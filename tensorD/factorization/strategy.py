# Created by ay27 at 17/2/23
import tensorflow as tf
import tensorD.loss as ltp
import tensorD.base.type as type
from tensorD.base.barrier import Barrier
from tensorD.base.logger import create_logger
import numpy as np

log = create_logger()


class Strategy(object):
    def create_graph(self, cluster):
        raise NotImplementedError

    def train(self, sess, feed_data):
        raise NotImplementedError

    def sync(self, sess, feed_data):
        raise NotImplementedError


class PPTTF(Strategy):
    def __init__(self, task_cnt, task_index, shape, R, lamb, tao, rho):
        self.grads = []
        self.cluster = None
        self.task_cnt = task_cnt
        self.task_index = task_index
        self.shape = shape
        self.R = R

        self.lamb = lamb
        self.tao = tao
        self.rho = rho

        self.supervisor = None
        self.global_step = None
        self.train_op = None
        self.sync_op = None
        self.loss_op = None
        self.summary_op = None
        self.X = None
        self.summary_writer = None
        self.grad = None

    def create_graph(self, cluster):
        with tf.device('/job:ps/task:0'):
            # self.barrier = Barrier(tf.get_default_session(), self.task_cnt)
            # self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.global_step = tf.get_variable('global_step', (), initializer=tf.constant_initializer(0),
                                               trainable=False)

        # for i in range(self.task_cnt):
        with tf.device('/job:worker/task:%d' % self.task_index):
            log.debug('task %d, create graph' % self.task_index)

            self.X = tf.placeholder(tf.float64, shape=self.shape)
            localA = tf.get_variable("A-%d" % self.task_index, shape=(self.shape[0], self.R), dtype=tf.float64,
                                     initializer=tf.random_uniform_initializer(minval=0.0, maxval=1.0,
                                                                               dtype=tf.float64))
            localB = tf.get_variable("B-%d" % self.task_index, shape=(self.shape[1], self.R), dtype=tf.float64,
                                     initializer=tf.random_uniform_initializer(minval=0.0, maxval=1.0,
                                                                               dtype=tf.float64))
            localC = tf.get_variable("C-%d" % self.task_index, shape=(self.shape[2], self.R), dtype=tf.float64,
                                     initializer=tf.random_uniform_initializer(minval=0.0, maxval=1.0,
                                                                               dtype=tf.float64))

            ktensor = type.KTensor([localA, localB, localC])

            self.loss_op = ltp.rmse(self.X - ktensor.extract())

            self.train_op = tf.train.GradientDescentOptimizer(0.002)

            self.grad = self.train_op.compute_gradients(self.loss_op, [localA, localB, localC])

            self.grads.append(self.grad)
            average_grad = self.average_grad()
            update_grad = self.train_op.apply_gradients(average_grad, global_step=self.global_step)
            self.sync_op = update_grad

        saver = tf.train.Saver()
        tf.summary.histogram('loss', self.loss_op)
        self.summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter('/tmp/ppttf', tf.get_default_graph())
        self.init_op = tf.global_variables_initializer()

        self.supervisor = tf.train.Supervisor(is_chief=(self.task_index == 0),
                                              init_op=self.init_op,
                                              summary_op=self.summary_op,
                                              saver=saver,
                                              global_step=self.global_step,
                                              save_model_secs=60)

    def train(self, sess, feed_data):
        log.debug('train begin')
        log.debug('train, feed size shape' + str(np.shape(feed_data)))
        loss_v, _, step = sess.run([self.loss_op, self.grad, self.global_step], feed_dict={self.X: feed_data})
        print('step = %d, loss = %f' % (step, loss_v))
        log.debug('train finish in step %d, worker %d' % (step, self.task_index))
        summary_str = sess.run(self.summary_op, feed_dict={self.X: feed_data})
        log.debug('summary finish')
        self.summary_writer.add_summary(summary_str)
        return loss_v

    def sync(self, sess, feed_data):
        log.debug('task %d, begin sync barrier' % self.task_index)
        log.debug('task %d, end of barrier' % self.task_index)
        sess.run(self.sync_op, feed_dict={self.X: feed_data})

    # copy from https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py
    def average_grad(self):
        average_grads = []

        for grad_and_vars in zip(*self.grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                grads.append(expanded_g)

            if tf.__version__.startswith('0.'):
                grad = tf.concat(0, grads)
            else:
                grad = tf.concat(grads, axis=0)
            grad = tf.reduce_mean(grad, axis=0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

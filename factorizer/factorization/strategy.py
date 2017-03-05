# Created by ay27 at 17/2/23
import tensorflow as tf
import factorizer.loss_template as ltp
import factorizer.base.type as type
from factorizer.base.barrier import Barrier
from factorizer.base.logger import create_logger

log = create_logger()


class Strategy(object):
    def create_graph(self, cluster):
        raise NotImplementedError

    def train(self, sess, feed_data):
        raise NotImplementedError

    def sync(self, sess):
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
        self.barrier = None
        self.loss_op = None
        self.summary_op = None
        self.X = None
        self.summary_writer = None

    def create_graph(self, cluster):
        with tf.device('/job:ps/task:0'):
            self.barrier = Barrier(tf.get_default_session(), self.task_cnt)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

        for i in range(self.task_cnt):
            with tf.device('/job:worker/task:%d' % i):
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

                self.loss_op = ltp.l2(self.X, ktensor.extract())

                self.train_op = tf.train.GradientDescentOptimizer(0.0002)

                grad = self.train_op.compute_gradients(self.loss_op, [localA, localB, localC])

                self.grads.append(grad)
                average_grad = self.average_grad()
                update_grad = self.train_op.apply_gradients(average_grad, global_step=self.global_step)
                self.sync_op = update_grad

        saver = tf.train.Saver()
        tf.summary.histogram('loss', self.loss_op)
        self.summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter('/tmp/ppttf', tf.get_default_graph())
        init_op = tf.global_variables_initializer()

        self.supervisor = tf.train.Supervisor(is_chief=(self.task_index == 0),
                                              init_op=init_op,
                                              summary_op=self.summary_op,
                                              saver=saver,
                                              global_step=self.global_step,
                                              save_model_secs=60)

    def train(self, sess, feed_data):
        loss_v, _, step = sess.run([self.loss_op, self.train_op, self.global_step], feed_data={self.X: feed_data})
        summary_str = sess.run(self.summary_op, feed_data={self.X: feed_data})
        self.summary_writer.add_summary(summary_str)
        return loss_v

    def sync(self, sess):
        log('begin sync barrier')
        self.barrier()
        log('end of barrier')
        sess.run(self.sync_op)

    def average_grad(self):
        res = tf.reduce_mean(self.grads)
        self.grads = []  # clean grads list
        return res

# Created by ay27 at 17/2/23
import tensorflow as tf
import factorizer.loss_template as ltp
import factorizer.base.type as type
from factorizer.base.barrier import Barrier


class Strategy(object):
    def create_graph(self, cluster):
        raise NotImplementedError

    def train(self, sess, feed_data):
        raise NotImplementedError


class PPTTF(Strategy):
    def __init__(self, task_cnt, task_index, order, lamb, tao, rho):
        self.cluster = None
        self.task_cnt = task_cnt
        self.task_index = task_index
        self.order = order

        self.lamb = lamb
        self.tao = tao
        self.rho = rho

        self.supervisor = None
        self.global_step = None
        self.train_op = None

    def create_graph(self, cluster):
        with tf.device('/job:ps/task:0'):
            globalA = tf.get_variable("gA", dtype=tf.float64)
            globalB = tf.get_variable("gB", dtype=tf.float64)
            globalC = tf.get_variable("gC", dtype=tf.float64)
            barrier = Barrier(tf.get_default_session(), self.task_cnt)

        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % self.task_index,
                cluster=cluster)):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

            X = tf.placeholder(tf.float64)
            localA = tf.get_variable("A", dtype=tf.float64)
            localB = tf.get_variable("B", dtype=tf.float64)
            localC = tf.get_variable("C", dtype=tf.float64)

            ktensor = type.KTensor([localA, localB, localC])

            self.loss_op = ltp.l2(X, ktensor.extract())

            grad = tf.gradients(self.loss_op, [localA, localB, localC])

            new_A = localA - self.tao * grad[0]
            new_B = localB - self.tao * grad[1]
            new_C = localC - self.tao * grad[2]

            assign_op1 = tf.assign(localA, new_A)
            assign_op2 = tf.assign(localB, new_B)
            assign_op3 = tf.assign(localC, new_C)



            self.train_op = tf.group(assign_op1, assign_op2, assign_op3)

            saver = tf.train.Saver()
            tf.summary.histogram('loss', self.loss_op)
            summary_op = tf.summary.merge_all()
            init_op = tf.global_variables_initializer()

        self.supervisor = tf.train.Supervisor(is_chief=(self.task_index == 0),
                                              init_op=init_op,
                                              summary_op=summary_op,
                                              saver=saver,
                                              global_step=self.global_step,
                                              save_model_secs=60)

    def train(self, sess, feed_data):
        loss_v, _, step = sess.run([self.loss_op, self.train_op, self.global_step])
        return loss_v

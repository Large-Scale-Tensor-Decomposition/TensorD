# Created by ay27 at 17/2/23
import tensorflow as tf
import factorizer.loss_template as ltp


class Strategy(object):
    def prepare(self):
        raise NotImplementedError

    def update(self, feed_data):
        raise NotImplementedError

    def sync(self):
        raise NotImplementedError


class PPTTF(Strategy):
    def __init__(self, cluster, task_cnt, task_index, order, lamb, tao, rho, tol=10e-4):
        self.cluster = cluster
        self.task_cnt = task_cnt
        self.task_index = task_index
        self.order = order
        self.tol = tol

        self.lamb = lamb
        self.tao = tao
        self.rho = rho

        self.supervisor = None
        self.global_step = None

    def prepare(self):
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % self.task_index,
                cluster=self.cluster)):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

            X = tf.placeholder(tf.float64)
            A = tf.get_variable("A", dtype=tf.float64)
            B = tf.get_variable("B", dtype=tf.float64)
            C = tf.get_variable("C", dtype=tf.float64)

            loss = ltp.l2(X, )

            new_A = A -

            saver = tf.train.Saver()
            tf.summary.histogram('loss', loss_value)
            summary_op = tf.summary.merge_all()
            init_op = tf.global_variables_initializer()

        self.supervisor = tf.train.Supervisor(is_chief=(self.task_index == 0),
                                              init_op=init_op,
                                              summary_op=summary_op,
                                              saver=saver,
                                              global_step=self.global_step,
                                              save_model_secs=60)

    def update(self, feed_data):
        pass

    def sync(self):
        pass

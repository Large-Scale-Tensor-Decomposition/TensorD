# Created by ay27 at 17/2/23
import tensorflow as tf


class Strategy(object):
    def prepare(self):
        raise NotImplementedError

    def update(self, feed_data):
        raise NotImplementedError

    def sync(self):
        raise NotImplementedError


class PPTTF(Strategy):
    def __init__(self, cluster, task_cnt, task_index, order, tol=10e-4):
        self.cluster = cluster
        self.task_cnt = task_cnt
        self.task_index = task_index
        self.order = order
        self.tol = tol
        self.supervisor = None

    def prepare(self):
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % self.task_index,
                cluster=self.cluster)):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

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
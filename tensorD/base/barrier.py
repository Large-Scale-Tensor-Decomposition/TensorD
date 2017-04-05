# Created by ay27 at 17/3/1
import tensorflow as tf
import time
from tensorD.base.logger import create_logger

logger = create_logger()


class Barrier(object):
    def __init__(self, task_cnt, sleep_interval=0.1):
        self.task_cnt = task_cnt
        self.sleep_interval = sleep_interval

        # there exist bug in zeros_initializer with dtype given
        # the initializer use constant_initializer, not zeros_initializer
        with tf.device('/job:ps/task:0'):
            self.counter1 = tf.get_variable(name='counter1', shape=(), dtype=tf.int32,
                                            initializer=tf.constant_initializer(0, dtype=tf.int32))
            self.counter2 = tf.get_variable(name='counter2', shape=(), dtype=tf.int32,
                                            initializer=tf.constant_initializer(0, dtype=tf.int32))
            self.add_op1 = self.counter1.assign_add(1, use_locking=True)
            self.add_op2 = self.counter2.assign_add(1, use_locking=True)

    def __call__(self, sess):
        sess.run(self.add_op1)
        logger.debug('counter = %d' % sess.run(self.counter1))
        while sess.run(self.counter1) % self.task_cnt != 0:
            time.sleep(self.sleep_interval)
        sess.run(self.add_op2)
        while sess.run(self.counter2) % self.task_cnt != 0:
            time.sleep(self.sleep_interval)

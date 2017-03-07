# Created by ay27 at 17/3/1
import tensorflow as tf
import time


class Barrier(object):
    def __init__(self, task_cnt, sleep_interval=0.1):
        self.task_cnt = task_cnt
        self.sleep_interval = sleep_interval

        # TODO: pull newest version tensorflow to fix the initializer bug
        self.counter1 = tf.get_variable(name='counter1', shape=[1], dtype=tf.int32, initializer=tf.zeros_initializer)
        self.counter2 = tf.get_variable(name='counter2', shape=[1], dtype=tf.int32, initializer=tf.zeros_initializer)
        self.add_op1 = self.counter1.assign_add(1, use_locking=True)
        self.add_op2 = self.counter2.assign_add(1, use_locking=True)

    def __call__(self, sess):
        sess.run(self.add_op1)
        while sess.run(self.counter1) % self.task_cnt != 0:
            time.sleep(self.sleep_interval)
        sess.run(self.add_op2)
        while sess.run(self.counter2) % self.task_cnt != 0:
            time.sleep(self.sleep_interval)

# Created by ay27 at 17/3/1
import tensorflow as tf
import time


class Barrier(object):
    def __init__(self, sess, task_cnt, sleep_interval=0.1):
        self.sess = sess
        self.task_cnt = task_cnt
        self.sleep_interval = sleep_interval

        self.counter1 = tf.get_variable('counter1', (), dtype=tf.int32, initializer=tf.zeros_initializer)
        self.counter2 = tf.get_variable('counter2', (), dtype=tf.int32, initializer=tf.zeros_initializer)
        self.add_op1 = self.counter1.assign_add(1, use_locking=True)
        self.add_op2 = self.counter2.assign_add(1, use_locking=True)

    def __call__(self, *args, **kwargs):
        self.sess.run(self.add_op1)
        while self.sess.run(self.counter1) % self.task_cnt != 0:
            time.sleep(self.sleep_interval)
        self.sess.run(self.add_op2)
        while self.sess.run(self.counter2) % self.task_cnt != 0:
            time.sleep(self.sleep_interval)

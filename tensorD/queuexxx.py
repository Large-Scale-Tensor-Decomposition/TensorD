#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/8/4 AM9:58
# @Author  : Shiloh Leung
# @Site    : 
# @File    : queuexxx.py
# @Software: PyCharm Community Edition


import tensorflow as tf
import numpy as np


def queue_test1():
    q = tf.FIFOQueue(3, 'float')
    init = q.enqueue_many(([0., 0., 0.],))

    x = q.dequeue()
    y = x + 1
    q_inc = q.enqueue([y])

    sess = tf.Session()
    init_v = sess.run(init)
    sess.run(q_inc)
    sess.run(q_inc)
    sess.run(q_inc)
    sess.run(q_inc)


queue_test1()
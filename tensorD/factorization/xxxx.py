#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/8/23 PM11:03
# @Author  : Shiloh Leung
# @Site    : 
# @File    : xxxx.py
# @Software: PyCharm Community Edition
import numpy as np
from tensorD.base import ops
import tensorflow as tf
from tensorD.DataBag import *
from functools import reduce
#
# for i in range(6):
#     X = tf.nn.relu(tf.random_normal(shape=(2,2)))
#     init_op = tf.global_variables_initializer()
#     sess = tf.Session()
#     sess.run(init_op)
#     print('%d time:' % (i+1))
#     print(sess.run(X))

number_list = [tf.Variable(i) for i in range(1, 9)]
prod = reduce(lambda a, b: a * b, number_list)
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)
print(sess.run(prod))
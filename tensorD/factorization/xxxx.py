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

# X = np.zeros((3, 4))
# X = np.array([[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]])
# T = tf.constant(X)
# X1 = ops.vectorize(T)
# X2 = ops.vec_to_tensor(X1, (3,4))
# sess = tf.Session()
# print(sess.run(T))
# print(sess.run(X1))
# print(sess.run(X2))
# X = np.arange(12).reshape(3, 4)
# T = tf.Variable(X, dtype=tf.float64)
# normT = tf.norm(T, ord='fro', axis=(0, 1))
# l = rand_list([3, 4, 5], 2)
# for mat in l:
#     print(mat)

X = np.array([[ 21239470.91245371,  14384532.32344297],[14384532.32344297, 17018956.13238818]])
T = tf.Variable(X, dtype=tf.float64)
normT = tf.reduce_max(tf.svd(T, compute_uv=False))
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)
print(sess.run(normT))


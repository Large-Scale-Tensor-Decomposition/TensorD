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

# number_list = [tf.Variable(i) for i in range(1, 9)]
# prod = reduce(lambda a, b: a * b, number_list)
# init_op = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init_op)
# print(sess.run(prod))
# np_X = np.array([[2, 0, 1], [-1, 1, 0], [-3, 3, 0]])
# np_matrices = [np_X for _ in range(3)]
# np_res = np.prod([max(np.linalg.svd(mat, compute_uv=0)) for mat in np_matrices])
# print('np result: ', np_res)
# tf_matrices = [tf.constant(np_X,dtype=tf.float64) for _ in range(3)]
# tf_res = ops.max_single_value_mul(tf_matrices)
# sess = tf.Session()
# print('tf result: ', sess.run(tf_res))
# core = gen_core(2)
# print(core[:,:,0])
# print(core[:,:,1])
# A = rand_list2([3,4,5], [2,2,2])
# print(A[0])
# print(A[1])
# print(A[2])
X = np.arange(60).reshape(3, 4, 5)
print(X[:,:,0])
print(X[:,:,1])
print(X[:,:,2])
print(X[:,:,3])
print(X[:,:,4])
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

X = np.zeros((3, 4))
X = np.array([[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]])
T = tf.constant(X)
X1 = ops.vectorize(T)
X2 = ops.vec_to_tensor(X1, (3,4))
sess = tf.Session()
print(sess.run(T))
print(sess.run(X1))
print(sess.run(X2))
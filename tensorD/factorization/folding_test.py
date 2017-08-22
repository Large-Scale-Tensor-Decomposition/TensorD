#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/8/22 PM3:22
# @Author  : Shiloh Leung
# @Site    : 
# @File    : folding_test.py
# @Software: PyCharm Community Edition
import tensorflow as tf
import numpy as np
from tensorD.loss import *
from tensorD.base import *
from numpy.random import rand

def fold_test():
    T = np.array([[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]], [[17, 18, 19, 20], [21, 22, 23, 24]]])
    print('origin tensor shape: \n', T.shape)
    print('\norigin tensor: \n', T)
    TT = tf.constant(T)
    X3 = ops.unfold(TT, 2)
    v3 = ops.vectorize(X3)
    v_m3 = ops.vec_to_tensor(v3, (4, 6))
    v_m_t3 = ops.fold(v_m3, 2, (3, 2, 4))
    sess = tf.Session()
    X3_res = sess.run(X3)
    v3_res = sess.run(v3)
    v_m3_res = sess.run(v_m3)
    v_m_t3_res = sess.run(v_m_t3)

    print('\nmode-3 matrix: ', X3.get_shape().as_list())
    print(X3_res)
    print('\nmode-3 vector: ', v3.get_shape().as_list())
    print(v3_res)
    print('\nback to mode-3 matrix: ', v_m3.get_shape().as_list())
    print(v_m3_res - X3_res)
    print('\nback to tensor: ', v_m_t3.get_shape().as_list())
    print(v_m_t3_res - T)



fold_test()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/2 PM4:59
# @Author  : Shiloh Leung
# @Site    : 
# @File    : DataGenerator.py
# @Software: PyCharm Community Edition


import numpy as np
from tensorD.base.type import KTensor
from tensorD.base.type import TTensor
import tensorflow as tf

def synthetic_data_cp(N_list, R):
    matrices = [None for _ in range(len(N_list))]
    for ii in range(len(N_list)):
        matrices[ii] = np.maximum(0, np.random.randn(N_list[ii], R))
    M = KTensor(matrices)
    with tf.Session() as sess:
        Mtrue = sess.run(M.extract())
    return Mtrue

def synthetic_data_tucker(N_list, Ranks):
    matrices = [None for _ in range(len(N_list))]
    G = np.maximum(0, np.random.randn(*Ranks))
    for ii in range(len(N_list)):
        matrices[ii] = np.random.randn(N_list[ii], Ranks[ii])
    M = TTensor(G, matrices)
    with tf.Session() as sess:
        Mtrue = sess.run(M.extract())
    return Mtrue

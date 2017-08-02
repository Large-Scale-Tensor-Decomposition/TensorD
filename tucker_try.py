#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/7/31 AM9:46
# @Author  : Shiloh Leung
# @Site    : 
# @File    : tucker_try.py
# @Software: PyCharm Community Edition
import numpy as np
import tensorflow as tf
from tensorD.base.type import TTensor
import tensorD.loss as loss
from numpy.random import rand
from tensorD.factorization.env import Environment
from tensorD.dataproc.provider import Provider
from tensorD.factorization.tucker import HOSVD
from tensorD.factorization.tucker import HOOI
from tensorD.DataBag import *



def my_hosvd_test(printinfo=False):
    data_provider = Provider()
    X_ = gen_test_tensor([3, 4, 5], 3)
    data_provider.full_tensor = lambda: tf.constant(X_, dtype=tf.float64)
    env = Environment(data_provider, summary_path='/tmp/tensord')
    hosvd = HOSVD(env)
    args = HOSVD.HOSVD_Args(ranks=[2,2,2])
    hosvd.build_model(args)
    hosvd.train()



    if printinfo == True:
        print("\nfactor matrices:")
        factors = hosvd.factors

        for matrix in factors:
            print(matrix)
        print("\ncore matrix:")
        core = hosvd.core
        print(core)

        P = TTensor(core, factors)
        print("\nfull tensor:")
        with tf.Session() as sess:
            seem_full = sess.run(P.extract())
        print(seem_full)
        print("\nreal full:")
        print(hosvd.full)

        print("\nsubtraction:")
        print(seem_full - hosvd.full)

        full = tf.constant(hosvd.full,dtype=tf.float64)
        input_tensor = tf.constant(X_, dtype=tf.float64)
        rmse = loss.rmse_ignore_zero(input_tensor, full)

        with tf.Session() as sess:
            rmse_v = sess.run(rmse)
        print("rmse=%.7f" % rmse_v)



def my_hooi_test(steps, printinfo=False):
    data_provider = Provider()
    X_ = gen_test_tensor([3, 4, 5], 3)
    data_provider.full_tensor = lambda: tf.constant(X_, dtype=tf.float64)
    env = Environment(data_provider, summary_path='/tmp/tensord')
    hooi = HOOI(env)
    args = hooi.HOOI_Args(ranks=[2, 2, 2])
    hooi.build_model(args)
    hooi.train(steps)

    if printinfo == True:
        print("\nfactor matrices:")
        factors = hooi.factors

        for matrix in factors:
            print(matrix)
        print("\ncore matrix:")
        core = hooi.core
        print(core)

        P = TTensor(core, factors)
        print("\nfull tensor:")
        with tf.Session() as sess:
            seem_full = sess.run(P.extract())
        print(seem_full)
        print("\nreal full:")
        print(hooi.full)

        print("\nsubtraction:")
        print(seem_full - hooi.full)

        full = tf.constant(hooi.full,dtype=tf.float64)
        input_tensor = tf.constant(X_, dtype=tf.float64)
        rmse = loss.rmse_ignore_zero(input_tensor, full)

        with tf.Session() as sess:
            rmse_v = sess.run(rmse)
        print("rmse=%.10f" % rmse_v)



#my_hosvd_test(True)
my_hooi_test(10)

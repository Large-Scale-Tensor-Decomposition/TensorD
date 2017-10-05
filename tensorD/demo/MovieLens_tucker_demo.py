#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/4 PM9:39
# @Author  : Shiloh Leung
# @Site    : 
# @File    : MovieLens_tucker_demo.py
# @Software: PyCharm Community Edition

from tensorD.dataproc.reader import TensorReader
import tensorflow as tf
from tensorD.factorization.env import Environment
from tensorD.dataproc.provider import Provider
from tensorD.factorization.tucker import HOOI
from tensorD.loss import *

if __name__ == '__main__':
    full_shape =  [671, 9066, 262]#[943, 1682, 31]
    # Train on *.base.csv
    print('=========Train=========')
    base = TensorReader('new_ratings_all.csv')#('movielens-100k/u1.base.csv')
    base.read(full_shape=full_shape)
    with tf.Session() as sess:
        rating_tensor = sess.run(base.full_data)
    data_provider = Provider()
    data_provider.full_tensor = lambda: rating_tensor
    env = Environment(data_provider, summary_path='/tmp/tucker_demo')
    hooi = HOOI(env)
    args = HOOI.HOOI_Args(ranks=[20, 20, 20], validation_internal=3)
    hooi.build_model(args)
    hooi.train(100)
    print('Train ends.\n\n\n')


    # Test on *.test.csv
    # print('=========Test=========')
    # test = TensorReader('movielens-100k/u1.test.csv')
    # test.read(full_shape=full_shape)
    # full = tf.constant(hooi.full, dtype=tf.float64)
    # rmse_op = rmse_ignore_zero(test.full_data, full)
    # with tf.Session() as sess:
    #     rmse = sess.run(rmse_op)
    # print('RMSE on u1.test.csv :  %.5f' % rmse)

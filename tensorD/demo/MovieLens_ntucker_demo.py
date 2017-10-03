#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/3 PM4:16
# @Author  : Shiloh Leung
# @Site    : 
# @File    : MovieLens_ntucker_demo.py
# @Software: PyCharm Community Edition

from tensorD.demo.MovieLens_process import *
from tensorD.dataproc.reader import TensorReader
import tensorflow as tf
from tensorD.factorization.env import Environment
from tensorD.dataproc.provider import Provider
from tensorD.factorization.ntucker import NTUCKER_BCU

if __name__ == '__main__':
    csv_file = 'movielens_data.csv'
    #movie_dict = data_write(csv_file)
    reader = TensorReader(csv_file)
    read_start = time.time()
    reader.read()
    read_end = time.time()
    print('Read time: %.6f s\n' % (read_end - read_start))
    data_start = time.time()
    with tf.Session() as sess:
        rating_tensor = sess.run(reader.full_data)    # (671, 9066, 262)
    data_end = time.time()
    print('Data process time: %.6f s\n' % (data_end - data_start))
    data_provider = Provider()
    data_provider.full_tensor = lambda: rating_tensor
    env = Environment(data_provider, summary_path='/tmp/ntucker_demo')
    ntucker = NTUCKER_BCU(env)
    args = NTUCKER_BCU.NTUCKER_Args(ranks=[100, 100, 100], validation_internal=5)
    ntucker.build_model(args)
    train_start = time.time()
    ntucker.train(10000)
    train_end = time.time()
    print('Training time: %.6f s\n' % (train_end - train_start))


#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/29 PM8:31
# @Author  : Shiloh Leung
# @Site    :
# @File    : ml_tucker.py
# @Software: PyCharm Community Edition


from tensorD.dataproc.reader import TensorReader
from tensorD.factorization.env import Environment
from tensorD.dataproc.provider import Provider
from tensorD.factorization.tucker import HOOI
from tensorD.factorization.tucker import HOSVD
from tensorD.demo.DataGenerator import *

if __name__ == '__main__':
    full_shape = [943, 1682, 31]
    base = TensorReader('/root/tensorD_f/data_out_tmp/u1.base.csv')
    base.read(full_shape=full_shape)
    with tf.Session() as sess:
        rating_tensor = sess.run(base.full_data)
    data_provider = Provider()
    data_provider.full_tensor = lambda: rating_tensor
    env = Environment(data_provider, summary_path='/tmp/tucker_ml')
    #hooi = HOOI(env)
    #args = HOOI.HOOI_Args(ranks=[20, 20, 20], validation_internal=1)
    #hooi.build_model(args)
    #hist = hooi.train(100)
    # out_path = '/root/tensorD_f/data_out_tmp/python_out/hooi_ml_20.txt'
    # with open(out_path, 'w') as out:
    #     for loss in hist:
    #         out.write('%.6f\n' % loss)

    hosvd = HOSVD(env)
    args = HOSVD.HOSVD_Args(ranks=[20, 20, 20])
    hosvd.build_model(args)
    hosvd.train()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 6/22/19 7:44 PM
# @Author  : Siqi Liang
# @Contact : zszxlsq@gmail.com
# @File    : sparse_problem.py
# @Software: PyCharm
import tensorflow as tf
import numpy as np
from tensorD.dataproc.reader import TensorReader
from tensorD.factorization.env import Environment
from tensorD.dataproc.provider import Provider
from tensorD.factorization.tucker import HOOI
from tensorD.factorization.tucker import HOSVD
from tensorD.base.type import TTensor

if __name__ == "__main__":
    with tf.Session() as sess:
        a = tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1.0, 2.0], dense_shape=[3, 4])
        a = tf.sparse.to_dense(a)
        data_provider = Provider()
        data_provider.full_tensor = lambda: a.eval(session=sess)
        env = Environment(data_provider, summary_path='/tmp/hosvd_' + '20', session=sess)
        hosvd = HOSVD(env)
        args = HOSVD.HOSVD_Args(ranks=[2, 2])
        hosvd.build_model(args)
        hosvd.train()
        factors = hosvd.factors
        core = hosvd.core
        recovered_a = TTensor(core=core, factors=factors)
        print(sess.run(recovered_a.extract()))



#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 5/5/19 4:28 PM
# @Author  : Siqi Liang
# @Contact : zszxlsq@gmail.com
# @File    : error_file.py
# @Software: PyCharm

from tensorD.factorization.env import Environment
from tensorD.dataproc.provider import Provider
from tensorD.factorization.tucker import HOOI
from tensorD.demo.DataGenerator import *

if __name__ == '__main__':
    for i in range(3):
        g1 = tf.Graph()
        data_provider = Provider()
        X = np.arange(60).reshape(3, 4, 5) + i
        data_provider.full_tensor = lambda: X
        hooi_env = Environment(data_provider, summary_path='/tmp/tensord')
        hooi = HOOI(hooi_env)
        args = hooi.HOOI_Args(ranks=[2, 2, 2], validation_internal=5)
        with g1.as_default() as g:
            hooi.build_model(args)
            hooi.train(100)
        print(np.sum(hooi.full - X))
        tf.reset_default_graph()

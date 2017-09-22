#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/8/23 PM4:12
# @Author  : Shiloh Leung
# @Site    : 
# @File    : ntucker_test.py
# @Software: PyCharm Community

import numpy as np
import tensorflow as tf
from numpy.random import rand
from tensorD.factorization.env import Environment
from tensorD.dataproc.provider import Provider
from tensorD.factorization.ntucker import NTUCKER_ALS
from  tensorD.DataBag import  *

if __name__ == '__main__':
    data_provider = Provider()
    X = np.arange(24).reshape(2, 3, 4) + 1
    data_provider.full_tensor = lambda: tf.constant(X, dtype=tf.float64)
    env = Environment(data_provider, summary_path='/tmp/tensord')
    ntucker = NTUCKER_ALS(env)
    args = NTUCKER_ALS.NTUCKER_Args(ranks=[2, 2, 2], validation_internal=5)
    ntucker.build_model(args)
    ntucker.train(100)


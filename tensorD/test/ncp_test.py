#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/9/12 PM7:57
# @Author  : Shiloh Leung
# @Site    : 
# @File    : ncp_test.py
# @Software: PyCharm Community Edition

from tensorD.factorization.env import Environment
from tensorD.factorization.ncp import NCP
from tensorD.dataproc.provider import Provider
import tensorflow as tf
import numpy as np
from tensorD.DataBag import *

if __name__ == '__main__':
    data_provider = Provider()
    X = gen_test_tensor([3, 4, 5], 3)
    data_provider.full_tensor = lambda: tf.constant(X, dtype=tf.float64)
    env = Environment(data_provider, summary_path='/tmp/tensord')
    ncp = NCP(env)
    args = NCP.NCP_Args(rank=2, validation_internal=1)
    ncp.build_model(args)
    ncp.train(4)
    #print(ncp.full - X)


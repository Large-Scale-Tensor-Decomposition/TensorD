#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/8/16 AM11:48
# @Author  : Shiloh Leung
# @Site    : 
# @File    : ntucker.py
# @Software: PyCharm Community Edition

import tensorflow as tf
import numpy as np
from tensorD.loss import *
from tensorD.base import *
from numpy.random import rand
from .factorization import BaseFact
from .env import Environment


class NTUCKER_ALS(BaseFact):
    class NTUCKER_Args(object):
        def __init__(self, ranks: list):
            self.ranks = ranks

    def __init__(self, env):
        assert isinstance(env, Environment)
        self._env = env
        self._model = None
        self._full_tensor = None
        self._factors = None
        self._core = None
        self._args = None
        self._init_op = None
        self._core_op = None
        self._factor_update_op = None
        self._is_train_finish = False

    def build_model(self, args):
        assert isinstance(args, NTUCKER_ALS.NTUCKER_Args)
        input_data = self._env.full_data()
        shape = input_data.get_shape().as_list()
        order = input_data.get_shape().ndims

        with tf.name_scope('random-initial') as scope:
            A = [tf.Variable(rand(shape[ii], args.ranks[ii]), name='A-%d' % ii) for ii in range(order)]
            g = tf.Variable(np.zeros(args.ranks))
            g_init_op = g.assign(ops.ttm(input_data, A, transpose=True))
        with tf.name_scope('unfold-all-mode') as scope:
            mats = [ops.unfold(input_data, mode) for mode in range(order)]
            assign_op = [None for _ in range(order)]

        for mode in range(order):
            if mode != 0:
                with tf.control_dependencies([assign_op[mode - 1]]):
                    S = ops.kron(A, mode, True)
                    GS_pinv = tf.py_func(np.linalg.pinv, [tf.matmul(ops.unfold(g, mode), S, name='GS-%d' % mode)], tf.float64, name='pinvGS-%d' % mode)
                    XGS_pinv = tf.matmul(mats[mode], GS_pinv, name='XGS-%d' % mode)
                    assign_op[mode] = A[mode].assign(tf.nn.relu(XGS_pinv))
                    SA_pinv = tf.py_func(np.linalg.pinv, [ops.kron([tf.transpose(S), assign_op[mode]])], tf.float64, name='pinvSA-%d' % mode)
                    vecG = tf.nn.relu(tf.matmul(SA_pinv, ops.vectorize(mats[mode])), name='vecG-%d' % mode)
                    # TODO : reshape core tensor g


    def predict(self, *key):
        if not self._full_tensor:
            raise TensorErr('improper stage to call predict before the model is trained')
        return self._full_tensor.item(key)

    @property
    def factors(self):
        return self._factors

    @property
    def core(self):
        return self._core

    @property
    def train_finish(self):
        return self._is_train_finish

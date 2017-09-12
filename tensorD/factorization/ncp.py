#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/8/16 AM11:48
# @Author  : Shiloh Leung
# @Site    : 
# @File    : ncp.py
# @Software: PyCharm Community Edition

import tensorflow as tf
import numpy as np
from numpy.random import rand
from tensorD.loss import *
from tensorD.DataBag import *
from tensorD.base import *
from .factorization import *
from .env import *

class NCP(BaseFact):
    class NCP_Args(object):
        def __init__(self, rank, validation_internal=-1, verbose=False):
            self.rank = rank
            self.validation_internal = validation_internal
            self.verbose = verbose

    def __init__(self, env):
        assert isinstance(env, Environment)
        self._env = env
        self._model = None
        self._full_tensor = None
        self._factors = None
        self._args = None
        self._init_op = None
        self._factor_update_op = None
        self._full_op = None
        self._loss_op = None
        self._is_train_finish = False

    def build_model(self, args):
        assert isinstance(args, NCP.NCP_Args)
        input_data = self._env.full_data()
        input_norm = tf.norm(input_data)
        shape = input_data.get_shape().as_list()
        size = np.prod(shape)
        order = len(shape)

        with tf.name_scope('random-initial') as scope:
            # TODO : best way to initialize is normally distributed pseudorandom numbers
            #A = [tf.Variable(tf.nn.relu(tf.random_normal([shape[ii], args.rank])), name='A-%d' % ii) for ii in range(order)]
            Uinit = rand_list(shape, args.rank)
            A = [tf.Variable(Uinit[ii], name='A-%d' % ii) for ii in range(order)]
            Am = A
            A0 = A
            A_update_op = [None for _ in range(order)]
            Am_update_op = [None for _ in range(order)]
            A0_update_op = [None for _ in range(order)]
            t0 = tf.Variable(1.0)
            t = tf.Variable(1.0)
            t_update_op = None
            t0_update_op = None
            wA = [tf.Variable(1.0) for _ in range(order)]
            wA_update_op = [None for _ in range(order)]
            L = [tf.Variable(1.0, name='gradientLipschitz-%d' % ii) for ii in range(order)]
            L0 = L
            L_update_op = [None for _ in range(order)]
            L0_update_op = [None for _ in range(order)]

        with tf.name_scope('normalize-initial') as scope:
            norm_init_op = [None for _ in range(order)]
            for mode in range(order):
                norm_init_op[mode] = A[mode].assign(A[mode]/tf.norm(A[mode], ord='fro')*input_norm)
        with tf.name_scope('unfold-all-mode') as scope:
            mats = [ops.unfold(input_data, mode) for mode in range(order)]
            sizeN = [int(size/In) for In in shape]


        t_update_op = t.assign((1+tf.sqrt(1+4*tf.square(t0)))/2)
        w = (t0 - 1)/t_update_op
        for mode in range(order):
            L0_update_op[mode] = L0[mode].assign(L[mode])
            if mode != 0:
                with tf.control_dependencies([A_update_op[mode - 1]]):
                    AtA = [tf.matmul(A[ii], A[ii], transpose_a=True, name='AtA-%d-%d' % (mode, ii)) for ii in range(order)]
            else:
                AtA = [tf.matmul(A[ii], A[ii], transpose_a=True, name='AtA-%d-%d' % (mode, ii)) for ii in range(order)]
            V = ops.hadamard(AtA, skip_matrices_index=mode)
            L_update_op[mode] = L[mode].assign(tf.norm(V))
            XA = tf.matmul(mats[mode], ops.khatri(A, mode, True), name='XA-%d' % mode)
            Gn = tf.subtract(tf.matmul(Am[mode], V), XA, name='G-%d' % mode)
            A_update_op[mode] = A[mode].assign(tf.nn.relu(tf.subtract(Am[mode], tf.div(Gn, L_update_op[mode]))))
            wA_update_op[mode] = wA[mode].assign(tf.minimum(w, tf.sqrt(L0_update_op[mode]/L_update_op[mode])))
            Am_update_op[mode] = Am[mode].assign(A_update_op[mode] + tf.matmul(wA_update_op[mode], A_update_op[mode]-A0[mode]))
            A0_update_op[mode] = A0[mode].assign(A_update_op[mode])

        t0_update_op = t0.assign(t_update_op)

        with tf.name_scope('full-tensor') as scope:
            P = KTensor(A)
            full_op = P.extract()
        with tf.name_scope('loss') as scope:
            loss_op = rmse_ignore_zero(input_data, full_op)














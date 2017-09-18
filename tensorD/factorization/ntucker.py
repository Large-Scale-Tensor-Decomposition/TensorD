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
from functools import reduce
from .factorization import BaseFact
from .env import Environment
from tensorD.DataBag import *

class NTUCKER_ALS(BaseFact):
    class NTUCKER_Args(object):
        def __init__(self, ranks: list, validation_internal=-1, verbose=False, tol=1.0e-4):
            self.ranks = ranks
            self.validation_internal = validation_internal
            self.verbose = verbose
            self.tol = tol

    def __init__(self, env):
        assert isinstance(env, Environment)
        self._env = env
        self._model = None
        self._full_tensor = None
        self._factors = None
        self._core = None
        self._args = None
        self._init_op = None
        self._core_init = None
        self._core_op = None
        self._factor_update_op = None
        self._full_op = None
        self._loss_op = None
        self._is_train_finish = False

    def build_model(self, args):
        assert isinstance(args, NTUCKER_ALS.NTUCKER_Args)
        input_data = self._env.full_data()
        input_norm = tf.norm(input_data)
        shape = input_data.get_shape().as_list()
        order = input_data.get_shape().ndims

        with tf.name_scope('random-init') as scope:
            # initialize with normally distributed pseudorandom numbers
            #A = [tf.Variable(tf.nn.relu(tf.random_normal(shape=(shape[ii], args.ranks[ii]), dtype=tf.float64)), name='A-%d' % ii, dtype=tf.float64) for ii in range(order)]
            # TODO : fix initialization in test
            Uinit = rand_list2(shape, args.ranks)
            A = [tf.Variable(Uinit[ii], name='A-%d' % ii) for ii in range(order)]
            A_update_op = [None for _ in range(order)]

        Am = [tf.Variable(np.zeros(shape=(shape[ii], args.ranks[ii])), dtype=tf.float64, name='Am-%d' % ii) for ii in range(order)]
        A0 = [tf.Variable(np.zeros(shape=(shape[ii], args.ranks[ii])), dtype=tf.float64, name='A0-%d' % ii) for ii in range(order)]

        with tf.name_scope('norm-init') as scope:
            norm_init_op = [None for _ in range(order)]
            Am_init_op = [None for _ in range(order)]
            A0_init_op = [None for _ in range(order)]
            for mode in range(order):
                norm_init_op[mode] = A[mode].assign(
                    A[mode] / tf.norm(A[mode], ord='fro', axis=(0, 1)) * tf.pow(input_norm, 1 / (order+1)))
                A0_init_op[mode] = A0[mode].assign(norm_init_op[mode])
                Am_init_op[mode] = Am[mode].assign(norm_init_op[mode])

        with tf.name_scope('core-init') as scope:
            # initialize with normally distributed pseudorandom numbers
            #g = tf.Variable(tf.nn.relu(tf.random_normal(shape=args.ranks, dtype=tf.float64)), name='core-tensor')
            g = tf.Variable(gen_core(args.ranks[0]), dtype=tf.float64, name='core-tensor')
            g_norm_init = g.assign(g / tf.norm(g) * tf.pow(input_norm, 1 / (order+1)))

        g0 = tf.Variable(np.zeros(shape=args.ranks), dtype=tf.float64, name='core_0')
        gm = tf.Variable(np.zeros(shape=args.ranks), dtype=tf.float64, name='core_m')
        g0_init_op = g0.assign(g_norm_init)
        gm_init_op = gm.assign(g_norm_init)

        t0 = tf.Variable(1.0, dtype=tf.float64, name='t0')
        t = tf.Variable(1.0, dtype=tf.float64, name='t')
        wA = [tf.Variable(1.0, dtype=tf.float64, name='wA-%d' % ii) for ii in range(order+1)]
        wA_update_op = [None for _ in range(order+1)]
        L = [tf.Variable(1.0, name='Lipschitz-%d' % ii, dtype=tf.float64) for ii in range(order+1)]
        L0 = [tf.Variable(1.0, name='Lipschitz0-%d' % ii, dtype=tf.float64) for ii in range(order+1)]
        L_update_op = [None for _ in range(order+1)]
        L0_update_op = [None for _ in range(order+1)]


        with tf.name_scope('unfold-all-mode') as scope:
            mats = [ops.unfold(input_data, mode) for mode in range(order)]


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

    def train(self, steps):
        sess = self._env.sess
        args = self._args

        init_op = self._init_op
        full_op = self._full_op
        factor_update_op = self._factor_update_op
        loss_op = self._loss_op

        sum_op = tf.summary.merge_all()
        sum_writer = tf.summary.FileWriter(self._env.summary_path, sess.graph)

        sess.run(init_op)

        print('Non-negative tucker model initial finish')
        for step in range(1, steps + 1):
            if (step == steps) or args.verbose or (step == 1) or (
                        step % args.validation_internal == 0 and args.validation_internal != -1):
                sum_msg = sess.run([sum_op])
                sum_writer.add_summary(sum_msg, step)
                print('step=%d, RMSE=%.10f' % (step, loss_v))
            else:
                self._factors = sess.run([])

        print('Non-negative tucker model train finish, with RMSE = %.10f' % loss_v)
        self._is_train_finish = True

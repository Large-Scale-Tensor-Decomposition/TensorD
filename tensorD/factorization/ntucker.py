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


class NTUCKER_ALS(BaseFact):
    class NTUCKER_Args(object):
        def __init__(self, ranks: list, validation_internal=-1, verbose=False):
            self.ranks = ranks
            self.validation_internal = validation_internal
            self.verbose = verbose

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
                    # TODO : update part 1
            else:
                # TODO : update part 1

            # TODO : update part 2

        # TODO : not sure whether could use in this way
        with tf.name_scope('full-tensor') as scope:
            P = TTensor(g_update_op, assign_op)
            full_op = P.extract()

        with tf.name_scope('loss') as scope:
            loss_op = rmse_ignore_zero(input_data, full_op)
        tf.summary.scalar('loss', loss_op)

        init_op = tf.global_variables_initializer()

        self._args = args
        self._init_op = init_op
        self._core_init = g_init_op
        self._full_op = full_op
        self._factor_update_op = assign_op
        self._core_op = g_update_op
        self._loss_op = loss_op

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
        g_init_op = self._core_init
        full_op = self._full_op
        factor_update_op = self._factor_update_op
        core_op = self._core_op
        loss_op = self._loss_op

        sum_op = tf.summary.merge_all()
        sum_writer = tf.summary.FileWriter(self._env.summary_path, sess.graph)

        sess.run(init_op)

        print('Non-negative tucker model initial finish')
        for step in range(1, steps + 1):
            if (step == steps) or args.verbose or (step == 1) or (step % args.validation_internal == 0 and args.validation_internal != -1):
                loss_v, self._full_tensor, self._factors, self._core, sum_msg = sess.run([loss_op, full_op, factor_update_op, core_op, sum_op])
                sum_writer.add_summary(sum_msg, step)
                print('step=%d, RMSE=%.15f' % (step, loss_v))
            else:
                self._factors, self._core = sess.run([factor_update_op, core_op])

        print('Non-negative tucker model train finish, with RMSE = %.15f' % loss_v)
        self._is_train_finish = True


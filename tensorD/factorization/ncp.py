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
        self._lambdas = None
        self._args = None
        self._init_op = None
        self._other_init_op = None
        self._train_op = None
        self._factor_update_op = None
        self._full_op = None
        self._loss_op = None
        self._is_train_finish = False

    def predict(self, *key):
        if not self._full_tensor:
            raise TensorErr('improper stage to call predict before the model is trained')
        return self._full_tensor.item(key)

    @property
    def full(self):
        return self._full_tensor

    @property
    def train_finish(self):
        return self._is_train_finish

    @property
    def factors(self):
        return self._factors

    @property
    def lambdas(self):
        return self._lambdas

    def build_model(self, args):
        assert isinstance(args, NCP.NCP_Args)
        input_data = self._env.full_data()
        input_norm = tf.norm(input_data)
        shape = input_data.get_shape().as_list()
        order = len(shape)

        with tf.name_scope('random-initial') as scope:
            # TODO : best way to initialize is normally distributed pseudorandom numbers
            #A = [tf.Variable(tf.nn.relu(tf.random_normal([shape[ii], args.rank])), name='A-%d' % ii) for ii in range(order)]
            Uinit = rand_list(shape, args.rank)
            A = [tf.Variable(Uinit[ii], name='A-%d' % ii) for ii in range(order)]
            Am = [tf.Variable(np.zeros(shape=(shape[ii], args.rank)), dtype=tf.float64) for ii in range(order)]
            A0 = [tf.Variable(np.zeros(shape=(shape[ii], args.rank)), dtype=tf.float64) for ii in range(order)]
            Am_init_op = [None for _ in range(order)]
            A0_init_op = [None for _ in range(order)]
            A_update_op = [None for _ in range(order)]
            Am_update_op = [None for _ in range(order)]
            A0_update_op = [None for _ in range(order)]
            t0 = tf.Variable(1.0, dtype=tf.float64)
            t = tf.Variable(1.0, dtype=tf.float64)
            t_update_op = None
            t0_update_op = None
            wA = [tf.Variable(1.0, dtype=tf.float64) for _ in range(order)]
            wA_update_op = [None for _ in range(order)]
            L = [tf.Variable(1.0, name='gradientLipschitz-%d' % ii, dtype=tf.float64) for ii in range(order)]
            L0 = [tf.Variable(1.0, dtype=tf.float64) for _ in range(order)]
            L_update_op = [None for _ in range(order)]
            L0_update_op = [None for _ in range(order)]

        with tf.name_scope('normalize-initial') as scope:
            norm_init_op = [None for _ in range(order)]
            for mode in range(order):
                norm_init_op[mode] = A[mode].assign(A[mode]/tf.norm(A[mode], ord='fro', axis=(0, 1))*tf.pow(input_norm, 1/order))
                A0_init_op[mode] = A0[mode].assign(norm_init_op[mode])
                Am_init_op[mode] = Am[mode].assign(norm_init_op[mode])
        with tf.name_scope('unfold-all-mode') as scope:
            mats = [ops.unfold(input_data, mode) for mode in range(order)]


        t_update_op = t.assign((1+tf.sqrt(1+4*tf.square(t0)))/2)
        w = (t0 - 1)/t_update_op
        for mode in range(order):

            if mode != 0:
                with tf.control_dependencies([A_update_op[mode - 1]]):
                    AtA = [tf.matmul(A[ii], A[ii], transpose_a=True, name='AtA-%d-%d' % (mode, ii)) for ii in range(order)]
                    XA = tf.matmul(mats[mode], ops.khatri(A, mode, True), name='XA-%d' % mode)
            else:
                AtA = [tf.matmul(A[ii], A[ii], transpose_a=True, name='AtA-%d-%d' % (mode, ii)) for ii in range(order)]
                XA = tf.matmul(mats[mode], ops.khatri(A, mode, True), name='XA-%d' % mode)
            V = ops.hadamard(AtA, skip_matrices_index=mode)
            L0_update_op[mode] = L0[mode].assign(L[mode])
            L_update_op[mode] = L[mode].assign(tf.reduce_max(tf.svd(V, compute_uv=False)))
            Gn = tf.subtract(tf.matmul(Am[mode], V), XA, name='G-%d' % mode)
            A_update_op[mode] = A[mode].assign(tf.nn.relu(tf.subtract(Am[mode], tf.div(Gn, L_update_op[mode]))))
            wA_update_op[mode] = wA[mode].assign(tf.minimum(w, tf.sqrt(L0_update_op[mode]/L_update_op[mode])))
            Am_update_op[mode] = Am[mode].assign(A_update_op[mode] + wA_update_op[mode]*(A_update_op[mode]-A0[mode]))
            with tf.control_dependencies([Am_update_op[mode]]):
                # add control dependencies to avoid random computation sequence
                A0_update_op[mode] = A0[mode].assign(A_update_op[mode])

        t0_update_op = t0.assign(t_update_op)

        with tf.name_scope('full-tensor') as scope:
            P = KTensor(A_update_op)
            full_op = P.extract()
        with tf.name_scope('loss') as scope:
            loss_op = rmse_ignore_zero(input_data, full_op)

        tf.summary.scalar('loss', loss_op)

        init_op = tf.global_variables_initializer()

        self._args = args
        self._init_op = init_op
        self._other_init_op = tf.group(*norm_init_op, *Am_init_op, *A0_init_op)
        self._train_op = tf.group(*L_update_op, *L0_update_op, t_update_op, t0_update_op, *Am_update_op, *wA_update_op, *A0_update_op)
        self._factor_update_op = A_update_op
        self._full_op = full_op
        self._loss_op = loss_op



    def train(self, steps):
        self._is_train_finish = False

        sess = self._env.sess
        args = self._args

        init_op = self._init_op
        other_init_op = self._other_init_op
        factor_update_op = self._factor_update_op
        train_op = self._train_op
        full_op = self._full_op
        loss_op = self._loss_op

        sum_op = tf.summary.merge_all()
        sum_writer = tf.summary.FileWriter(self._env.summary_path, sess.graph)

        sess.run(init_op)
        sess.run(other_init_op)
        print('Non-Negative CP model initial finish')

        for step in range(1, steps + 1):
            if (step == steps) or (args.verbose) or (step == 1) or (step % args.validation_internal == 0 and args.validation_internal != -1):
                self._factors, self._full_tensor, loss_v, sum_msg,  _ = sess.run([factor_update_op, full_op, loss_op, sum_op, train_op])
                sum_writer.add_summary(sum_msg, step)
                print('step=%d, RMSE=%f' % (step, loss_v))
            else:
                self._factors, _ = sess.run([factor_update_op, train_op])
        self._lambdas = np.ones(shape=(1, args.rank))

        print('Non-Negative CP model train finish, with RMSE = %.10f' % (loss_v))
        self._is_train_finish = True











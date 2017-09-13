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
        self._norm_init_op = None
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
        #insert_test(input_norm)
        shape = input_data.get_shape().as_list()
        size = np.prod(shape)
        order = len(shape)

        with tf.name_scope('random-initial') as scope:
            # TODO : best way to initialize is normally distributed pseudorandom numbers
            #A = [tf.Variable(tf.nn.relu(tf.random_normal([shape[ii], args.rank])), name='A-%d' % ii) for ii in range(order)]
            Uinit = rand_list(shape, args.rank)
            A = [tf.Variable(Uinit[ii], name='A-%d' % ii) for ii in range(order)]
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
            L0 = L
            L_update_op = [None for _ in range(order)]
            L0_update_op = [None for _ in range(order)]

        with tf.name_scope('normalize-initial') as scope:
            norm_init_op = [None for _ in range(order)]
            for mode in range(order):
                norm_init_op[mode] = A[mode].assign(A[mode]/tf.norm(A[mode], ord='fro', axis=(0, 1))*tf.pow(input_norm, 1/order))
        #insert_test(norm_init_op, 2)
        A0 = A
        Am = A
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
                    XA = tf.matmul(mats[mode], ops.khatri(A, mode, True), name='XA-%d' % mode)
                    # A_res, AtA_res = sess.run([A_update_op[0], AtA])
            else:
                AtA = [tf.matmul(A[ii], A[ii], transpose_a=True, name='AtA-%d-%d' % (mode, ii)) for ii in range(order)]
                XA = tf.matmul(mats[mode], ops.khatri(A, mode, True), name='XA-%d' % mode)
            V = ops.hadamard(AtA, skip_matrices_index=mode)
            L_update_op[mode] = L[mode].assign(tf.reduce_max(tf.svd(V, compute_uv=False)))
            Gn = tf.subtract(tf.matmul(Am[mode], V), XA, name='G-%d' % mode)
            A_update_op[mode] = A[mode].assign(tf.nn.relu(tf.subtract(Am[mode], tf.div(Gn, L_update_op[mode]))))
            wA_update_op[mode] = wA[mode].assign(tf.minimum(w, tf.sqrt(L[mode]/L_update_op[mode])))
            Am_update_op[mode] = Am[mode].assign(A_update_op[mode] + wA_update_op[mode]*(A_update_op[mode]-A0[mode]))
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
        self._norm_init_op = norm_init_op
        self._full_op = full_op
        self._L_update_op = L_update_op
        self._L0_update_op = L0_update_op
        self._t_update_op = t_update_op
        self._t0_update_op = t0_update_op
        self._factor_update_op = A_update_op
        self._wA_update_op = wA_update_op
        self._Am_update_op = Am_update_op
        self._A0_update_op = A0_update_op
        self._loss_op = loss_op


    def train(self, steps):
        self._is_train_finish = False

        sess = self._env.sess
        args = self._args
        init_op = self._init_op
        norm_init_op = self._norm_init_op
        full_op = self._full_op
        factor_update_op = self._factor_update_op
        wA_update_op = self._wA_update_op
        Am_update_op = self._Am_update_op
        A0_update_op = self._A0_update_op
        L_update_op = self._L_update_op
        L0_update_op = self._L0_update_op
        t_update_op = self._t_update_op
        t0_update_op = self._t0_update_op
        loss_op = self._loss_op

        sum_op = tf.summary.merge_all()
        sum_writer = tf.summary.FileWriter(self._env.summary_path, sess.graph)

        sess.run(init_op)
        sess.run(norm_init_op)
        print('Non-Negative CP model initial finish')

        for step in range(1, steps + 1):
            if (step == steps) or (args.verbose) or (step == 1) or (step % args.validation_internal == 0 and args.validation_internal != -1):
                self._factors, wA, Am, A0, self._full_tensor, loss_v, t, t0, L, L0, sum_msg = sess.run([factor_update_op, wA_update_op, Am_update_op, A0_update_op, full_op, loss_op, t_update_op, t0_update_op, L_update_op, L0_update_op, sum_op])
                print('factors:')
                for mat1 in self._factors:
                    print(mat1)
                #
                # print('\nAm:')
                # for mat2 in Am:
                #     print(mat2)
                #
                # print('\nA0:')
                # for mat3 in A0:
                #     print(mat3)


                # print('wA:')
                # print(wA)
                # print('t:')
                # print(t)
                # print('t0:')
                # print(t0)
                # print('L:')
                # print(L)
                # print('L0:')
                # print(L0)

                sum_writer.add_summary(sum_msg, step)
                print('step=%d, RMSE=%f' % (step, loss_v))
            else:
                self._factors, wA, Am, A0, t, t0, L, L0 = sess.run([factor_update_op, wA_update_op, Am_update_op, A0_update_op, t_update_op, t0_update_op, L_update_op, L0_update_op])
        self._lambdas = np.ones(shape=(1,args.rank))
        print('Non-Negative CP model train finish, with RMSE = %.10f' % (loss_v))
        self._is_train_finish = True











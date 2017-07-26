#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/7/24 PM6:35
# @Author  : Shiloh Leung
# @Site    : 
# @File    : cp_xxx.py
# @Software: PyCharm Community Edition
# Created by ay27 at 17/1/13
import numpy as np
import tensorflow as tf
from tensorD.base import *
from tensorD.loss import *
from numpy.random import rand
from .factorization import Model, BaseFact
from .env import Environment
from DataBag import *


class CP_ALS(BaseFact):
    class CP_Args(object):
        def __init__(self,
                     rank,
                     tol=10e-6,
                     validation_internal=-1,
                     get_lambda=False,
                     get_rmse=False,
                     verbose=False):
            self.rank = rank
            self.tol = tol
            self.validation_internal = validation_internal
            self.get_lambda = get_lambda
            self.get_rmse = get_rmse
            self.verbose = verbose

    def __init__(self, env):
        assert isinstance(env, Environment)
        self._env = env
        self._model = None
        self._full_tensor = None
        self._is_train_finish = False
        self._lambdas = None



    def predict(self, *key):
        if not self._full_tensor:
            raise TensorErr('improper stage to call predict before the model is trained')
        return self._full_tensor.item(key)

    def train(self, steps=None):
        self._train_in_single(steps)

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
        assert isinstance(args, CP_ALS.CP_Args)
        input_data = self._env.full_data()
        shape = input_data.get_shape().as_list()
        order = len(shape)
        # TODO : Fix the initial random matrices ?
        with tf.name_scope('random-initial') as scope:
            #A = [tf.Variable(rand(shape[ii], args.rank), name='A-%d' % ii) for ii in range(order)]
            Uinit = rand_list(shape, args.rank)
            #A = [tf.Variable(Uinit[ii], name='A_1_iter-%d' % ii) for ii in range(order)]
            A0 = tf.Variable(Uinit[0],name='A0')
            A1 = tf.Variable(Uinit[1], name='A1')
            A2 = tf.Variable(Uinit[2], name='A2')
            mats = [ops.unfold(input_data, mode) for mode in range(order)]

            assign_op_iter1 = [None for _ in range(order)]
            assign_op = [None for _ in range(order)]

        # TODO : iter 1 is special
        AtA_0 = [tf.matmul(A0,A0,transpose_a=True), tf.matmul(A1,A1,transpose_a=True), tf.matmul(A2,A2,transpose_a=True)]
        XA_0 = tf.matmul(mats[0], ops.khatri([A0,A1,A2], 0, True),name='XA0')
        V0 = ops.hadamard(AtA_0, skip_matrices_index=0)
        non_norm_A_0 = tf.matmul(XA_0, tf.py_func(np.linalg.pinv, [V0], tf.float64),name='non-norm0')
        lambda_op_0 = tf.norm(tf.reshape(non_norm_A_0,shape=(shape[0], args.rank)), axis=0,name='lambda0')
        assign_op0 = A0.assign(tf.div(non_norm_A_0, lambda_op_0))
        #insert_test(lambda_op_0)

        with tf.control_dependencies([assign_op0]):
            AtA_1 = [tf.matmul(A0, A0, transpose_a=True), tf.matmul(A1, A1, transpose_a=True), tf.matmul(A2, A2, transpose_a=True)]
            XA_1 = tf.matmul(mats[1], ops.khatri([A0, A1, A2], 1, True),name='XA1')
        V1 = ops.hadamard(AtA_1, skip_matrices_index=1)
        non_norm_A_1 = tf.matmul(XA_1, tf.py_func(np.linalg.pinv, [V1], tf.float64),name='non-norm1')
        lambda_op_1 = tf.norm(tf.reshape(non_norm_A_1, shape=(shape[1], args.rank)), axis=0,name='lambda1')
        assign_op1 = A1.assign(tf.div(A1, lambda_op_1))
        #insert_test(lambda_op_1)


        with tf.control_dependencies([assign_op1]):
            AtA_2 = [tf.matmul(A0, A0, transpose_a=True), tf.matmul(A1, A1, transpose_a=True), tf.matmul(A2, A2, transpose_a=True)]
            XA_2 = tf.matmul(mats[2], ops.khatri([A0, A1, A2], 2, True),name='XA2')
        V2 = ops.hadamard(AtA_2, skip_matrices_index=2)
        non_norm_A_2 = tf.matmul(XA_2, tf.py_func(np.linalg.pinv, [V2], tf.float64),name='non-norm2')
        lambda_op_2 = tf.norm(tf.reshape(non_norm_A_2, shape=(shape[2], args.rank)), axis=0,name='lambda2')
        assign_op2 = A2.assign(tf.div(A2, lambda_op_2))
        #insert_test(lambda_op_2)

        init_op = tf.global_variables_initializer()

        before_train = [self._env, args, init_op]
        in_train = [assign_op0, assign_op1, assign_op2, lambda_op_0, lambda_op_1, lambda_op_2]

        self._model = Model(before_train, in_train)
        return self._model

    def _train_in_single(self, steps):
        self._is_train_finish = False
        self._full_tensor = None

        sess = self._env.sess
        model = self._model
        args = model.before_train[1]
        init_op = model.before_train[2]

        assign_op0 = model.in_train[0]
        assign_op1 = model.in_train[1]
        assign_op2 = model.in_train[2]
        lambda_op0 = model.in_train[3]
        lambda_op1 = model.in_train[4]
        lambda_op2 = model.in_train[5]

        sum_op = tf.summary.merge_all()
        sum_writer = tf.summary.FileWriter(self._env.summary_path, sess.graph)

        sess.run(init_op)

        print('CP model initial finish')

        assign0, lambda0 = sess.run([assign_op0, lambda_op0])
        print("after assign 0:")
        print(assign0)
        print("lambda0:")
        print(lambda0)

        assign1, lambda1 = sess.run([assign_op1, lambda_op1])
        print("after assign 1:")
        print(assign1)
        print("lambda1:")
        print(lambda1)

        assign2, lambda2 = sess.run([assign_op2, lambda_op2])
        print("after assign 2:")
        print(assign2)
        print("lambda2:")
        print(lambda2)

        self._is_train_finish = True
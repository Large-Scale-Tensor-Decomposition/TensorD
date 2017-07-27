# Created by ay27 at 17/1/21

import tensorflow as tf
import numpy as np
from tensorD.loss import *
from tensorD.base import *
from .factorization import *
from .env import *


class HOSVD(BaseFact):
    class HOSVD_Args(object):
        def __init__(self, ranks: list):
            self.ranks = ranks

    def __init__(self, env: Environment):
        self._env = env
        self._model = None
        self._full_tensor = None
        self._is_train_finish = False

    def build_model(self, args) -> Model:
        input_data = self._env.full_data()
        order = input_data.get_shape().ndims
        A = []
        for n in range(order):
            _, U, _ = tf.svd(ops.unfold(input_data, n), full_matrices=True, name='svd-%d' % n)
            A.append(U[:, :args.ranks[n]])
        g = ops.ttm(input_data, A, transpose=True)

        P = TTensor(g, A)

        init_op = tf.global_variables_initializer()
        train_op = tf.group(g, *A)
        full_op = P.extract()
        loss_op = rmse_ignore_zero(input_data, full_op)
        var_list = A

        self._model = Model(self._env, train_op, loss_op, var_list, init_op, full_op, args)
        return self._model


    def train(self, steps=None):
        """

        Parameters
        ----------
        steps : Ignore

        Returns
        -------

        """
        self._is_train_finish = False
        self._full_tensor = None
        sess = self._env.sess
        model = self._model

        sess.run(model.init_op)
        print('HOSVD model initial finish')
        _, loss_v, self._full_tensor = sess.run([model.train_op, model.loss_op, model.full_tensor_op])
        self._factors = sess.run(model.var_list)
        g = ops.ttm(model.full_tensor_op, model.var_list, transpose=True)
        self._core = sess.run(g)
        print('HOSVD model train finish, with RMSE = %f' % loss_v)
        self._is_train_finish = True

    def full(self):
        return self._full_tensor

    def predict(self, *key):
        return self._full_tensor.item(key)

    def factors(self):
        return self._factors

    def core(self):
        return self._core

    def train_finish(self):
        return self._is_train_finish


class HOOI(BaseFact):
    class HOOI_Args(object):
        def __init__(self, ranks: list, validation_internal=-1, verbose=False):
            self.ranks = ranks
            self.validation_internal = validation_internal
            self.verbose = verbose

    def __init__(self, env: Environment):
        self._env = env
        self._model = None
        self._full_tensor = None
        self._is_train_finish = False

    def build_model(self, args) -> Model:
        assert isinstance(args, HOOI.HOOI_Args)
        return self._build_single_model(args)

    def train(self, steps=None):
        self._train_in_single(steps)

    def full(self):
        return self._full_tensor

    def predict(self, *key):
        return self._full_tensor.item(key)

    def factors(self):
        return self._factors

    def core(self):
        return self._core

    def train_finish(self):
        return self._is_train_finish


    def _build_single_model(self, args):
        input_data = self._env.full_data()
        shape = input_data.get_shape().as_list()
        order = input_data.get_shape().ndims

        # HOSVD to initialize factors A
        A = [tf.Variable(np.random.rand(shape[ii], args.ranks[ii]), name='A-%d' % ii) for ii in range(order)]
        init_ops = [None for _ in range(order)]
        for mode in range(order):
            _, U, _ = tf.svd(ops.unfold(input_data, mode), full_matrices=True, name='svd-%d' % mode)
            init_ops[mode] = A[mode].assign(U[:, :args.ranks[mode]])

        train_ops = [None for _ in range(order)]
        for mode in range(order):
            Y = ops.ttm(input_data, A, skip_matrices_index=mode, transpose=True)
            _, tmp, _ = tf.svd(ops.unfold(Y, mode))
            train_ops[mode] = A[mode].assign(tmp[:, :args.ranks[mode]])

        g = ops.ttm(input_data, A, transpose=True)

        init_op = tf.group(*init_ops)
        train_op = tf.group(*train_ops, g)
        P = TTensor(g, A)
        full_tensor_op = P.extract()
        loss_op = rmse_ignore_zero(input_data, full_tensor_op)
        var_list = A

        tf.summary.scalar('loss', loss_op)

        self._model = Model(self._env, train_op, loss_op, var_list, init_op, full_tensor_op, args)
        return self._model

    def _train_in_single(self, steps):
        sess = self._env.sess
        model = self._model
        args = model.args
        sum_op = tf.summary.merge_all()
        sum_writer = tf.summary.FileWriter(self._env.summary_path, sess.graph)

        sess.run(model.init_op)
        print('HOOI model initial finish')
        for step in range(steps):
            sess.run(model.train_op)
            if step + 1 == steps:
                loss_v, self._full_tensor = sess.run([model.loss_op, model.full_tensor_op])
                sum_writer.add_summary(sess.run(sum_op), step)
                self._factors = sess.run(model.var_list)
                g = ops.ttm(model.full_tensor_op, model.var_list, transpose=True)
                self._core = sess.run(g)
                print('step=%d, RMSE=%f' % (step, loss_v))
            elif args.verbose or step == 0 or step % args.validation_internal == 0:
                loss_v = sess.run(model.loss_op)
                sum_writer.add_summary(sess.run(sum_op), step)
                print('step=%d, RMSE=%f' % (step, loss_v))
        print('HOOI model train finish, with RMSE = %f' % loss_v)
        self._is_train_finish = True


# Created by ay27 at 17/1/13
import numpy as np
import tensorflow as tf
from tensorD.base.type import KTensor

import tensorD.base.ops as ops
from tensorD.loss import rmse
from numpy.random import rand
from tensorD.factorization.factorization import Model, BaseFact
from tensorD.factorization.env import Environment


class CP_ALS(BaseFact):
    class CP_Args(object):
        def __init__(self,
                     rank=20,
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
        self.env = env
        self.model = None
        self.full_tensor = None
        self.is_train_finish = False

    def build_model(self, args) -> Model:
        assert isinstance(args, CP_ALS.CP_Args)

        if self.env.is_distributed:
            # TODO
            pass
        else:
            input_data = self.env.full_data()
            shape = input_data.get_shape().as_list()
            order = len(shape)
            A = [tf.Variable(rand(shape[ii], args.rank), name='A-%d' % ii) for ii in range(order)]
            mats = [ops.unfold(input_data, mode) for mode in range(order)]

            assign_op = [None for _ in range(order)]
            for mode in range(order):
                AtA = [tf.matmul(A[ii], A[ii], transpose_a=True, name='AtA-%d' % ii) for ii in range(order)]
                V = ops.hadamard(AtA, skip_matrices_index=mode)
                # Unew
                XA = tf.matmul(mats[mode], ops.khatri(A, mode, True))
                assign_op[mode] = A[mode] = A[mode].assign(
                    tf.transpose(tf.matrix_solve(tf.transpose(V), tf.transpose(XA))))

            P = KTensor(A)
            full_op = P.extract()
            loss_op = rmse(input_data - full_op)

            tf.summary.scalar('loss', loss_op)

            train_op = tf.group(*assign_op)
            var_list = A

            init_op = tf.global_variables_initializer()

        self.model = Model(self.env, train_op, loss_op, var_list, init_op, full_op, args)
        return self.model

    def predict(self, key):
        pass

    def train(self, steps):
        self.is_train_finish = False
        self.full_tensor = None

        sess = self.env.sess
        model = self.model
        args = model.args

        sum_op = tf.summary.merge_all()
        sum_writer = tf.summary.FileWriter(self.env.summary_path, sess.graph)

        sess.run(model.init_op)

        print('CP model initial finish')
        for step in range(steps):
            sess.run(model.train_op)
            if step+1 == steps:
                loss_v, self.full_tensor = sess.run([model.loss_op, model.full_tensor_op])
                sum_writer.add_summary(sess.run(sum_op), step)
                print('step=%d, RMSE=%f' % (step, loss_v))

            elif args.verbose or step == 0 or step % args.validation_internal == 0:
                loss_v = sess.run(model.loss_op)
                sum_writer.add_summary(sess.run(sum_op), step)
                print('step=%d, RMSE=%f' % (step, loss_v))

        print('CP model train finish, with RMSE = %f' % loss_v)
        self.is_train_finish = True

    def full(self):
        return self.full_tensor


def cp(sess, tensor, rank, steps=100, tol=10e-4, ignore_tol=True, get_lambdas=False, get_rmse=False, verbose=False):
    shape = tensor.get_shape().as_list()
    order = len(shape)
    A = [tf.constant(rand(sp, rank)) for sp in shape]
    mats = [ops.unfold(tensor, mode) for mode in range(order)]

    sess.run(tf.global_variables_initializer())

    AtA = [tf.matmul(A[i], A[i], transpose_a=True) for i in range(order)]

    lambdas = None

    pre_rmse = 0.0

    for step in range(steps):
        for mode in range(order):
            V = ops.hadamard(AtA, skip_matrices_index=mode)
            # Unew
            XA = tf.matmul(mats[mode], ops.khatri(A, mode, True))
            # A[mode] = tf.matmul(XA, tf.matrix_inverse(V))
            A[mode] = tf.transpose(tf.matrix_solve(tf.transpose(V), tf.transpose(XA)))

            if get_lambdas:
                if step == 0:
                    lambdas = tf.sqrt(tf.reduce_sum(tf.square(A[mode]), 0))
                else:
                    lambdas = tf.maximum(tf.reduce_max(tf.abs(A[mode]), 0),
                                         tf.ones(A[mode].get_shape()[1].value, dtype=tf.float64))
                A[mode] = tf.map_fn(lambda x_row: x_row / lambdas, A[mode][:])

            AtA[mode] = tf.matmul(A[mode], A[mode], transpose_a=True)

        if not ignore_tol:
            P = KTensor(A, lambdas)
            res = sess.run(rmse(tensor - P.extract()))
            if step != 0 and abs(res - pre_rmse) < tol:
                return P
            pre_rmse = res

            if verbose:
                print(res)

    P = KTensor(A, lambdas)
    if get_rmse:
        res = rmse(tensor - P.extract())
        tf.summary.histogram('loss', res)
        op = tf.summary.merge_all()
        res = sess.run(res)

        tf.summary.FileWriter('/tmp/cp', sess.graph).add_summary(sess.run(op))

        return res, P
    else:
        return P


# Try to build the graph before run session, but failed.
# This function has errors!!!
def fake_cp(sess, tensor, rank, steps=100):
    shape = tensor.get_shape().as_list()
    order = len(shape)

    # graph = tf.get_default_graph()

    A = [tf.get_variable('A%d' % dim, shape=(dim, rank), dtype=tf.float64) for dim in shape]
    AtA = [tf.matmul(A[i], A[i], transpose_a=True) for i in range(order)]
    mats = [ops.unfold(tensor, _) for _ in range(order)]

    as_ops = list(range(order))

    for mode in range(order):
        V = ops.hadamard(AtA, skip_matrices_index=mode)
        XA = tf.matmul(mats[mode], ops.khatri(A, mode, True))
        with sess.graph.control_dependencies([V, XA]):
            tmp = tf.transpose(tf.matrix_solve(tf.transpose(V), tf.transpose(XA)))
        with sess.graph.control_dependencies([tmp]):
            as_ops[mode] = A[mode].assign(tmp)
            # A[mode] = tmp
        with sess.graph.control_dependencies([as_ops[mode]]):
            AtA[mode] = tf.matmul(A[mode], A[mode], transpose_a=True)
            tf.summary.histogram('AtA', AtA[mode])

    with sess.graph.control_dependencies(as_ops):
        P = KTensor(A)
        loss = rmse(tensor - P.extract())

    tf.summary.histogram('loss', loss)

    # e_step = tf.group(loss)

    merge_op = tf.summary.merge_all()
    sum_writer = tf.summary.FileWriter('/tmp/fake_cp', sess.graph)

    init = tf.global_variables_initializer()
    sess.run(init)

    for step in range(steps):
        # sess.run(loss)
        res = sess.run(loss)
        print('step %d, loss=%f' % (step, res))

        sum_str = sess.run(merge_op)
        sum_writer.add_summary(sum_str)
    print(sess.run(tensor - P.extract()))

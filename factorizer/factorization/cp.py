# Created by ay27 at 17/1/13
import numpy as np
import tensorflow as tf
from factorizer.base.type import KTensor

import factorizer.base.ops as ops
from factorizer.validator import rmse

rand = np.random.rand


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
        res = sess.run(rmse(tensor - P.extract()))
        return res, P
    else:
        return P

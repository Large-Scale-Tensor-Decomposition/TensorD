# Created by ay27 at 17/1/13
import tensorflow as tf
import numpy as np
import src.ops as ops
from src.type import KTensor
from src.validator import rmse
import time
rand = np.random.rand


def cp(sess, tensor, rank, steps=100, tol=10e-7, verbose=True):
    ts = time.time()
    shape = tensor.get_shape().as_list()
    order = len(shape)
    A = [tf.constant(rand(sp, rank)*10) for sp in shape]
    mats = [ops.unfold(tensor, mode) for mode in range(order)]

    for step in range(steps):
        for mode in range(order):
            AtA = [tf.matmul(A[i], A[i], transpose_a=True) for i in range(order)]
            V = ops.hadamard(AtA, skip_matrices_index=mode)
            XA = tf.matmul(mats[mode], ops.khatri(A, mode, True))
            A[mode] = tf.matmul(XA, tf.matrix_inverse(V))
            # A[mode] = tf.matrix_solve(V.T, XA.T).T

            # if step == 0:
            #     lambdas = tf.sqrt(tf.reduce_sum(tf.square(A[mode]), 0))
            # else:
            #     lambdas = tf.maximum(tf.reduce_max(A[mode], 0), tf.ones(A[mode].get_shape()[1].value, dtype=tf.float64))
            # A[mode] = tf.map_fn(lambda x_row: x_row / lambdas, A[mode][:])

            AtA[mode] = tf.matmul(A[mode], A[mode], transpose_a=True)

    P = KTensor(A)
    sess.run(tf.global_variables_initializer())
    res = rmse(tensor - P.extract())
    print(sess.run(res))

    print(time.time() - ts)

# Created by ay27 at 17/1/21

import tensorflow as tf
import numpy as np
from factorizer.validator import rmse
from factorizer.base import type
from factorizer.base import ops


def HOSVD(tensor, ranks):
    """

    :param tensor: tf.Tensor

    :param ranks: List

    :return: TTensor
    """
    order = tensor.get_shape().ndims
    A = []
    for n in range(order):
        _, U, _ = tf.svd(ops.unfold(tensor, n), full_matrices=True)
        A.append(U[:, :ranks[n]])
    g = ops.ttm(tensor, A, transpose=True)
    return g, A


def HOOI(tensor, ranks, steps=100, verbose=False):
    order = tensor.get_shape().ndims
    _, A = HOSVD(tensor, ranks)

    for step in range(steps):
        for n in range(order):
            Y = ops.ttm(tensor, A, skip_matrices_index=n, transpose=True)
            _, tmp, _ = tf.svd(ops.unfold(Y, n))
            A[n] = tmp[:, :ranks[n]]
        if verbose:
            g = ops.ttm(tensor, A, transpose=True)
            res = ops.ttm(g, A)
            err =rmse(tensor - res).eval()
            print('step %d, rmse=%f' % (step, err))
    g = ops.ttm(tensor, A, transpose=True)
    return type.TTensor(g, A)

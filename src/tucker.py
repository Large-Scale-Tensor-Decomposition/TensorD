from _operator import add
from functools import reduce
import math
import tensorflow as tf
import numpy as np
import src.ops as ops
from src.type import KTensor
from src.validator import rmse
import time

rand = np.random.rand

def RMSE(A, B):
    a = ops.vectorize(A)
    b = ops.vectorize(B)

    if A.shape != B.shape:
        raise ValueError('the shape of A and B must be equal')

    t = reduce(add, np.power(a - b, 2))

    return math.sqrt(t / len(a))



def HOSVD(tensor):

    tmp = list(range(len(tensor.shape)))
    l = range(0, len(tmp))
    X = list(l)  # matrix list
    U = list(l)  # left singular value list
    x = tensor
    for i in l:
        U[i], _, _ = np.linalg.svd(ops.unfold(tensor, i), True, True)
        x = ops.mul(x, U[i].T, i, i)

    return x, U


def HOOI(tensor, R=10, steps=100, tol=1e-10):

    tmp = list(range(len(tensor.shape)))
    # core = np.zeros(self.shape)
    l = range(len(tmp))
    A = list(l)
    x = tensor
    Ti = tensor
    _, U = HOSVD(tensor)
    # for iter_num in range(iter):
    for s in steps:
        for i in l:
            tmp.remove(i)
            for ii in range(len(tmp)):
                Ti = ops.mul(Ti, U[ii].T, ii, ii)
                A[i], _, _ = np.linalg.svd(ops.t2mat(i, tmp), True, True)
                # A[i] = (Y[i])[:, 0:lsr]
            tmp.append(i)
            tmp.sort()
            x = ops.mul(x, A[i].T, i, i)

        return  A, x



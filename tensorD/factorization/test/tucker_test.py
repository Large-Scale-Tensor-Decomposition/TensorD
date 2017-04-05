# Created by ay27 at 17/1/21
import unittest
import tensorflow as tf
import numpy as np
from tensorD.base import type
from tensorD.base import ops
from tensorD.factorization.tucker import HOSVD, HOOI
from tensorD.loss import rmse
import time
from numpy.random import rand
import logging

logger = logging.getLogger('TEST')


class MyTestCase(unittest.TestCase):
    def test_HOSVD(self):
        # x = np.arange(60).reshape(3, 4, 5)
        x = rand(3, 4, 5)

        with tf.Session().as_default():
            g, A = HOSVD(tf.constant(x, dtype=tf.float64), [3, 4, 5])
            res = type.TTensor(g, A).extract().eval()
        # print(res - x)

    def test_HOOI(self):

        ts = time.time()

        G = rand(15, 15, 15)
        A = rand(30, 15)
        B = rand(40, 15)
        C = rand(20, 15)

        np_X = np.einsum('xyz,ax,by,cz->abc', G, A, B, C)

        G = tf.constant(G)
        A = tf.constant(A)
        B = tf.constant(B)
        C = tf.constant(C)

        # 30x40x20
        with tf.Session().as_default():
            X = ops.ttm(G, [A, B, C])
            ttensor = HOOI(X, [15, 15, 15], steps=20)
            res = ttensor.extract().eval()
        np.testing.assert_array_almost_equal(np_X, res, 2)
        logger.info('HOOI : run time %f' % (time.time() - ts))


if __name__ == '__main__':
    unittest.main()

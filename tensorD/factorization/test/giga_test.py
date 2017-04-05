# Created by ay27 at 17/4/5
import time
import unittest

import numpy as np
import tensorflow as tf
import logging
from tensorD.factorization.giga import gigatensor
from numpy.random import rand

assert_array_equal = np.testing.assert_array_almost_equal

logger = logging.getLogger('TEST')


class MyTestCase(unittest.TestCase):
    def test_giga(self):
        I = 20
        J = 30
        K = 40
        R = 10

        X = rand(I,J,K)

        sess = tf.Session()
        gigatensor(sess, tf.constant(X, dtype=tf.float64))


if __name__ == '__main__':
    unittest.main()

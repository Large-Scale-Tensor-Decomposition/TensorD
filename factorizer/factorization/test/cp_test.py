# Created by ay27 at 17/1/17
import time
import unittest

import numpy as np
import tensorflow as tf
import logging
from factorizer.factorization.cp import cp
from numpy.random import rand
assert_array_equal = np.testing.assert_array_almost_equal


logger = logging.getLogger('TEST')


class MyTestCase(unittest.TestCase):
    def test_cp(self):
        # x = np.arange(60000).reshape(30, 40, 50)
        x = rand(30, 40, 50)

        ts = time.time()

        sess = tf.Session()
        with sess.as_default():
            res, _ = cp(sess, tf.constant(x, dtype=tf.float64), 20, steps=20, get_rmse=True)
            logger.info(res)

        logger.info(time.time() - ts)


if __name__ == '__main__':
    unittest.main()

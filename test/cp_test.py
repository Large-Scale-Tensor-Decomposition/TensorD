# Created by ay27 at 17/1/17
import unittest
import numpy as np
import tensorflow as tf
from src.cp import cp
import time

rand = np.random.rand
assert_array_equal = np.testing.assert_array_almost_equal


class MyTestCase(unittest.TestCase):
    def test_cp(self):
        # x = np.arange(60000).reshape(30, 40, 50)
        x = rand(30, 40, 50)

        ts = time.time()

        sess = tf.Session()
        with sess.as_default():
            res, _ = cp(sess, tf.constant(x, dtype=tf.float64), 20, steps=20, get_rmse=True)
            print(res)

        print(time.time() - ts)


if __name__ == '__main__':
    unittest.main()

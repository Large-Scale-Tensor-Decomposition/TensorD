# Created by ay27 at 17/1/11
import unittest
import tensorflow as tf
import tensorly as tly
import src.ops as ops
import numpy as np


class MyTestCase(unittest.TestCase):
    def test_khatri(self):
        a = np.random.rand(3, 4)
        b = np.random.rand(5, 4)
        c = np.random.rand(6, 4)
        with tf.Session().as_default():
            res = ops.khatri([a, b, c]).eval()
        tmp = np.einsum('az,bz,cz->abcz', a, b, c).reshape(-1, 4)
        np.testing.assert_array_almost_equal(res, tmp)

    def kron(self):
        a = np.random.rand(3, 4)
        b = np.random.rand(5, 6)
        c = np.kron(a, b)
        with tf.Session().as_default():
            res = ops.kron([a, b]).eval()
        np.testing.assert_array_almost_equal(res, c)

if __name__ == '__main__':
    unittest.main()

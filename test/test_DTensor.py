from unittest import TestCase
import numpy as np
import tensorflow as tf
import src.ops as ops
import src.type as tp
__author__ = 'Administrator'


class TestDTensor(TestCase):
    def setUp(self):
        self.tmpcase1 = [[[1, 13], [4, 16], [7, 19], [10, 22]],
                        [[2, 14], [5, 17], [8, 20], [11, 23]],
                        [[3, 15], [6, 18], [9, 21], [12, 24]]]
        self.tt = 0
        self.vectorcase1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                            13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        self.mc1 = np.rand(7, 8)
        self.mc2 = np.rand(100, 200)
        self.mc3 = np.rand(100, 200)
        self.matrices = [self.mc2, self.mc3]
        self.tmp = 0


    def test_mul(self):
        self.fail()

    def test_unfold(self):
        x = np.array(self.tmpcase1)
        r1 = np.reshape(np.transpose(self.tmpcase1, [1, 2, 0]), [4, 6])
        with tf.Session().as_default():
            r2 = ops.unfold(x, 1).eval()
        np.testing.assert_array_almost_equal(r1, r2)

    def test_t2mat(self):
        self.fail()

    def test_vectorize(self):
        self.fail()

    def test_get_shape(self):
        self.fail()

    def test_kron(self):
        self.fail()

    def test_fold(self):
        self.fail()
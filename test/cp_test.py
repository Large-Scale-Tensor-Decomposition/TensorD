# Created by ay27 at 17/1/12
import unittest
import tensorflow as tf
import numpy as np
from src.factorization import CP


class MyTestCase(unittest.TestCase):
    def test_cp(self):
        X = np.random.rand(20,30,15)
        sess = tf.Session()
        with sess.as_default():
            CP(sess, X, 10, learning_rate=0.002)

if __name__ == '__main__':
    unittest.main()

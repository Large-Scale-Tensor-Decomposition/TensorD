# Created by ay27 at 17/1/12
import unittest
import tensorflow as tf
import numpy as np
from src.factorization import CP_ALS
from src.cp import parafac


class MyTestCase(unittest.TestCase):
    def test_cp(self):
        X = np.random.rand(20,30,15)
        sess = tf.Session()
        parafac(X, 10, verbose=True)
        with sess.as_default():
            CP_ALS(sess, X, 15, steps=200, learning_rate=0.0002)

if __name__ == '__main__':
    unittest.main()

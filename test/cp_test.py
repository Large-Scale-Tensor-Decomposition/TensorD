# Created by ay27 at 17/1/17
import unittest
import numpy as np
import tensorflow as tf
from src.cp import cp

rand = np.random.rand
assert_array_equal = np.testing.assert_array_almost_equal


class MyTestCase(unittest.TestCase):
    def test_cp(self):
        x = rand(30,40,50)*10
        sess = tf.Session()
        with sess.as_default():
            cp(sess, tf.constant(x), 10, steps=20)


if __name__ == '__main__':
    unittest.main()

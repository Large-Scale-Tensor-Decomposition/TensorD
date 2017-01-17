# Created by ay27 at 17/1/17
import unittest
import numpy as np
import tensorflow as tf
from src.cp import cp

rand = np.random.rand
assert_array_equal = np.testing.assert_array_almost_equal


class MyTestCase(unittest.TestCase):
    def test_cp(self):
        x = rand(3,4,5)*10
        with tf.Session().as_default():
            cp(tf.constant(x), 3, steps=20)


if __name__ == '__main__':
    unittest.main()

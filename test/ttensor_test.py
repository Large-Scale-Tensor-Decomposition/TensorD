# Created by ay27 at 17/1/15
import unittest
from src.type import TTensor
import tensorflow as tf
import numpy as np

rand = np.random.rand


class MyTestCase(unittest.TestCase):
    def test_extract(self):
        g = rand(2, 3, 4)
        a = rand(2, 5)
        b = rand(3, 6)
        c = rand(4, 7)

        res1 = np.einsum('abc,ax,by,cz->xyz', g, a, b, c)

        tg = tf.constant(g)
        ta = tf.constant(a)
        tb = tf.constant(b)
        tc = tf.constant(c)

        with tf.Session().as_default():
            tt = TTensor(tg, [ta, tb, tc])
            res2 = tt.extract().eval()

        np.testing.assert_array_almost_equal(res1, res2)

if __name__ == '__main__':
    unittest.main()

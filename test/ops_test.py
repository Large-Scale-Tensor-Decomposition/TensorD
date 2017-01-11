# Created by ay27 at 17/1/11
import unittest
import tensorflow as tf
import numpy as np
import src.ops as ops


class MyTestCase(unittest.TestCase):
    def test_unfold(self):
        # 3x4x2
        tmp = [[[1, 13], [4, 16], [7, 19], [10, 22]],
               [[2, 14], [5, 17], [8, 20], [11, 23]],
               [[3, 15], [6, 18], [9, 21], [12, 24]]]
        x = np.array(tmp)

        r1 = np.reshape(np.transpose(tmp, [1, 2, 0]), [4, 6])
        with tf.Session().as_default():
            r2 = ops.unfold(x, 1).eval()
        np.testing.assert_array_almost_equal(r1, r2)

    def test_fold(self):
        tmp = [[[1, 13], [4, 16], [7, 19], [10, 22]],
               [[2, 14], [5, 17], [8, 20], [11, 23]],
               [[3, 15], [6, 18], [9, 21], [12, 24]]]
        x = np.array(tmp)
        with tf.Session().as_default():
            result = ops.fold(ops.unfold(x, 1), 1, (3, 4, 2)).eval()
        np.testing.assert_array_equal(x, result)


if __name__ == '__main__':
    unittest.main()

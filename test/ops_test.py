# Created by ay27 at 17/1/11
import unittest
import tensorflow as tf
import numpy as np
import src.ops as ops


class MyTestCase(unittest.TestCase):
    def setUp(self):
        # shape of tmp 3x4x2
        self.tmp = [[[1, 13], [4, 16], [7, 19], [10, 22]],
                    [[2, 14], [5, 17], [8, 20], [11, 23]],
                    [[3, 15], [6, 18], [9, 21], [12, 24]]]
        self.np_x = np.array(self.tmp)
        self.tf_x = tf.constant(self.np_x)

    def test_gen_perm(self):
        x = [2, 3, 1, 0]
        res = ops._gen_perm(4, 2)
        np.testing.assert_array_equal(x, res)

    def test_unfold(self):
        # mode 1: 3x4x2 -> 4x2x3 -> 4x6
        r1 = np.reshape(np.transpose(self.np_x, [1, 2, 0]), [4, 6])
        with tf.Session().as_default():
            r2 = ops.unfold(self.tf_x, 1).eval()
        np.testing.assert_array_almost_equal(r1, r2)

    def test_fold(self):
        # mode 2 : 3x4x2 -> 2x4x3 -> 2x12
        mode_2_mat = np.reshape(np.transpose(self.np_x, [2, 1, 0]), [2, 12])
        mode_2_tf = tf.constant(mode_2_mat)
        with tf.Session().as_default():
            res = ops.fold(mode_2_tf, 2, [3, 4, 2]).eval()
        np.testing.assert_array_almost_equal(self.np_x, res)

    def test_t2mat(self):
        np_A = np.random.rand(2, 3, 4, 5)
        tf_A = tf.constant(np_A)

        res1 = np.reshape(np.transpose(np_A, [1, 2, 3, 0]), [12, 10])
        with tf.Session().as_default():
            res2 = ops.t2mat(tf_A, [1, 2], [3, 0]).eval()
        np.testing.assert_array_almost_equal(res1, res2)

    def test_vectorize(self):
        res1 = np.reshape(self.np_x, -1)

        with tf.Session().as_default():
            res2 = ops.vectorize(self.tf_x).eval()
        np.testing.assert_array_equal(res1, res2)

    def test_vec_to_tensor(self):
        np_vec = np.reshape(self.np_x, -1)
        tf_vec = tf.constant(np_vec)
        with tf.Session().as_default():
            res = ops.vec_to_tensor(tf_vec, (3, 4, 2)).eval()
        np.testing.assert_array_almost_equal(self.np_x, res)

    def test_mul(self):
        np_A = np.random.rand(2, 3, 4)
        np_B = np.random.rand(4, 3, 6)
        np_res = np.einsum('ijk,klm->ijlm', np_A, np_B)

        tf_A = tf.constant(np_A)
        tf_B = tf.constant(np_B)
        with tf.Session().as_default():
            tf_res = ops.mul(tf_A, tf_B, [2], [0]).eval()
        self.assertEqual(len(np_res.shape), 4)
        np.testing.assert_array_almost_equal(np_res, tf_res)

    def test_inner(self):
        np_A = np.random.rand(2, 3, 4)
        np_B = np.random.rand(2, 3, 4)
        np_res = np.sum(np.reshape(np_A, -1) * np.reshape(np_B, -1))

        tf_A = tf.constant(np_A)
        tf_B = tf.constant(np_B)
        with tf.Session().as_default():
            tf_res = ops.inner(tf_A, tf_B).eval()
        np.testing.assert_almost_equal(np_res, tf_res)

    def test_kron(self):
        a = np.kron(self.mc2, self.mc3)
        b = np.kron(self.mc3, self.mc2)
        with tf.Session().as_default():
            r1 = ops.kron(self.matrices).eval()
            r2 = ops.kron(self.matrices, None, True).eval()
        np.testing.assert_array_almost_equal(a, r1)
        np.testing.assert_array_almost_equal(b, r2)

    def test_khatri(self):
        a = np.random.rand(3, 4)
        b = np.random.rand(5, 4)
        c = np.random.rand(6, 4)
        with tf.Session().as_default():
            res = ops.khatri([a, b, c]).eval()
        tmp = np.einsum('az,bz,cz->abcz', a, b, c).reshape(-1, 4)
        np.testing.assert_array_almost_equal(res, tmp)


if __name__ == '__main__':
    unittest.main()

# Created by ay27 at 17/1/11
import unittest
import tensorflow as tf
import numpy as np
import src.ops as ops


class MyTestCase(unittest.TestCase):
   def setUp(self):
        self.tmpcase1 = [[[1, 13], [4, 16], [7, 19], [10, 22]],
                        [[2, 14], [5, 17], [8, 20], [11, 23]],
                        [[3, 15], [6, 18], [9, 21], [12, 24]]]
        self.tt = 0

        self.vectorcase1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

        self.mc1 = np.random.rand(7, 8)
        self.mc2 = np.random.rand(100, 200)
        self.mc3 = np.random.rand(100, 200)
        self.matrices = [self.mc2, self.mc3]
        self.tmp = 0

   def test_unfold(self):
        # 3x4x2 tensor
        x = np.array(self.tmpcase1)

        r1 = np.reshape(np.transpose(self.tmpcase1, [1, 2, 0]), [4, 6])
        with tf.Session().as_default():
            r2 = ops.unfold(x, 1).eval()
        np.testing.assert_array_almost_equal(r1, r2)

   def test_fold(self):
        x = np.array(self.tmpcase1)
        with tf.Session().as_default():
            r = ops.fold(ops.unfold(x, 1), 1, (3, 4, 2)).eval()
        np.testing.assert_array_almost_equal(x, r)


    # n

   def test_t2mat(self):
        x = np.array(self.tmpcase1)
        r1 = np.reshape(np.transpose(self.tmpcase1, [1,2,0]),[4, 6])
        # set axis value 1.
        axis = 0
        matrix = np.array(self.tmpcase1)
        indices = list(range(len(matrix.shape)))
        indices[0], indices[axis] = indices[axis], indices[0]
        tmp = ops.matricization(axis, indices[1:])
        tmp = np.dot(matrix, tmp)
        # tmp = np.array(tmp)
        back_shape = [self.shape[_] for _ in indices]
        back_shape[0] = tmp.shape[0]
        np.reshape(tmp, back_shape).transpose(indices)
        with tf.Session().as_default():
            r2 = ops.t2mat(x, 1, [0, 2]).eval()
        np.testing.assert_array_almost_equal(r1, r2)

   def test_vectorize(self):
        x = np.array(self.tmpcase1)
        r1 = x.reshape(-1)

        with tf.Session().as_default():
            r2 = ops.vectorize(x).eval()
        np.testing.assert_array_equal(r1, r2)

   def test_vec_to_tensor(self):
        x = np.array(self.vectorcase1)
        r1 = np.reshape(x, (3, 4, 2))
        with tf.Session().as_default():
            r2 = ops.vec_to_tensor(x, (3, 4, 2))

   def test_dot(self):
        pass


   def test_inner(self):
        r1 = 0
        x1 = self.mc2.reshape(-1)
        x2 = self.mc3.reshape(-1)
        for i in range(len(x1)):
            r1 = r1 + x1[i] * x2[i]
        with tf.Session().as_default():
            r2 = ops.inner(self.mc2, self.mc3).eval()
        np.testing.assert_array_almost_equal(r1, r2)

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

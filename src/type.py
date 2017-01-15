# Created by ay27 at 17/1/11
import tensorflow as tf
import numpy as np
import src.ops as ops


class DTensor:
    """
    Dense Tensor
    """

    def __init__(self, tensor):
        """
        :param tensor: tf.Tensor or ndarray
        """
        if isinstance(tensor, tf.Tensor):
            self.T = tensor
            self.shape = tf.shape(tensor)
        else:
            self.T = tf.constant(tensor)
            self.shape = tensor.shape
        self.unfold_T = None
        self.fold_T = self.T

    def mul(self, tensor, a_axis, b_axis):
        return DTensor(ops.mul(self.T, tensor.T, a_axis, b_axis))

    def unfold(self, mode=0):
        if self.unfold_T is None:
            self.unfold_T = ops.unfold(self.T, mode)
        return DTensor(self.unfold_T)

    def t2mat(self, r_axis, c_axis):
        return DTensor(ops.t2mat(self.T, r_axis, c_axis))

    def vectorize(self):
        return DTensor(ops.vectorize(self.T))

    def get_shape(self):
        return self.T.get_shape()

    def kron(self, tensor):
        if isinstance(tensor, DTensor):
            return np.kron(self.T.eval(), tensor.T.eval())
        else:
            return np.kron(self.T.eval(), tensor)

    @staticmethod
    def fold(unfolded_tensor, mode, shape):
        return DTensor(ops.fold(unfolded_tensor, mode, shape))

    def __add__(self, other):
        if isinstance(other, DTensor):
            return DTensor(self.T + other.T)
        else:
            return DTensor(self.T + other)

    def __mul__(self, other):
        if isinstance(other, DTensor):
            return DTensor(self.T * other.T)
        else:
            return DTensor(self.T * other)

    def __sub__(self, other):
        if isinstance(other, DTensor):
            return DTensor(self.T - other.T)
        else:
            return DTensor(self.T - other)

    def __getitem__(self, index):
        return self.T[index]


class KTensor:
    """
    Kruskal Tensor
    """

    def __init__(self, factors, lambdas=None):
        self.U = factors
        if lambdas is None:
            self.lambdas = tf.ones(len(factors))
        else:
            self.lambdas = lambdas

    def extract(self):
        pass


class TTensor:
    """
    Tucker Tensor

    \mathcal{X} =  \mathcal{G} \times_1 \mathbf{A} \times_2 \mathbf{B} \times_3 \mathbf{C}

    """

    def __init__(self, core, factors):
        """
        construct the Tucker Tensor
        :param core: tf.Tensor, ndarray
        :param factors: List of tf.Tensor or ndarray
        """
        if isinstance(core, np.ndarray):
            self.g = tf.constant(core)
        else:
            self.g = core
        if isinstance(factors[0], np.ndarray):
            self.U = [tf.constant(mat) for mat in factors]
        else:
            self.U = factors
        self.order = self.g.get_shape().ndims

    def extract(self):
        """
        extract the full tensor of core and factors
        :return: tf.Tensor
        full tensor
        """
        g_start = ord('a')
        u_start = ord('z') - self.order + 1

        # construct the operator subscripts, such as: abc,ax,by,cz->xyz
        g_source = ''.join(chr(g_start + i) for i in range(self.order))
        u_source = ','.join(chr(g_start + i) + chr(u_start + i) for i in range(self.order))
        dest = ''.join(chr(u_start + i) for i in range(self.order))
        operator = g_source + ',' + u_source + '->' + dest
        return tf.einsum(operator, *([self.g] + self.U))

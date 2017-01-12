# Created by ay27 at 17/1/11
import tensorflow as tf
import numpy as np
import src.ops as ops


class TTensor:
    def __init__(self, tf_tensor):
        self.T = tf_tensor
        self.shape = tf_tensor.shape
        self.unfold_T = None
        self.fold_T = self.T

    def dot(self, tensor, a_axis, b_axis):
        return TTensor(ops.mul(self.T, tensor.T, a_axis, b_axis))

    def unfold(self, mode=0):
        if self.unfold_T is None:
            self.unfold_T = ops.unfold(self.T, mode)
        return TTensor(self.unfold_T)

    def t2mat(self, r_axis, c_axis):
        return TTensor(ops.t2mat(self.T, r_axis, c_axis))

    def vectorize(self):
        return TTensor(ops.vectorize(self.T))

    def get_shape(self):
        return self.T.get_shape()

    def kron(self, tensor):
        if isinstance(tensor, TTensor):
            return np.kron(self.T.eval(), tensor.T.eval())
        else:
            return np.kron(self.T.eval(), tensor)

    @staticmethod
    def fold(unfolded_tensor, mode, shape):
        return TTensor(ops.fold(unfolded_tensor, mode, shape))

    def __add__(self, other):
        if isinstance(other, TTensor):
            return TTensor(self.T + other.T)
        else:
            return TTensor(self.T + other)

    def __mul__(self, other):
        if isinstance(other, TTensor):
            return TTensor(self.T * other.T)
        else:
            return TTensor(self.T * other)

    def __sub__(self, other):
        if isinstance(other, TTensor):
            return TTensor(self.T - other.T)
        else:
            return TTensor(self.T - other)

    def __getitem__(self, index):
        return self.T[index]

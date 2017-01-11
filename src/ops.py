# Created by ay27 at 17/1/11
import tensorflow as tf
import numpy as np


def __gen_perm(order, mode):
    """
    Generate the specified permutation by the given mode
    """
    tmp = list(range(order - 1, -1, -1))
    tmp.remove(mode)
    perm = [mode] + tmp
    return perm


def unfold(tensor, mode=0):
    """
    Unfold tensor to a matrix, using Kolda-type
    :param tensor: tf.Tensor, ndarray
    :param mode: int, default is 0
    :return: tf.Tensor
    """
    perm = __gen_perm(len(tensor.shape), mode)
    return tf.reshape(tf.transpose(tensor, perm), (tensor.shape[mode], -1))


def fold(unfolded_tensor, mode, shape):
    """
    Fold an unfolded tensor to tensor with specified shape
    :param unfolded_tensor: tf.Tensor, ndarray
    :param mode: int
    :param shape: the specified shape of target tensor
    :return: tf.Tensor
    """
    perm = __gen_perm(len(shape), mode)
    shape_now = [shape[_] for _ in perm]
    back_perm = [item[0] for item in sorted(enumerate(perm), key=lambda x: x[1])]
    return tf.transpose(tf.reshape(unfolded_tensor, shape_now), back_perm)


def t2mat(tensor, r_axis, c_axis):
    """
    Transfer a tensor to a matrix by given row axis and column axis
    :param tensor: tf.Tensor, ndarray
    :param r_axis: int, list
    :param c_axis: int, list
    :return: matrix-like tf.Tensor
    """
    if isinstance(r_axis, int):
        indies = [r_axis]
        row_size = tensor.shape[r_axis]
    else:
        indies = r_axis
        row_size = np.prod([tensor.shape[i] for i in r_axis])
    if isinstance(c_axis, int):
        if c_axis == -1:
            c_axis = [_ for _ in range(len(tensor.shape)) if _ not in r_axis]
        indies.append(c_axis)
        col_size = tensor.shape[c_axis]
    else:
        indies = indies + c_axis
        col_size = np.prod([tensor.shape[i] for i in c_axis])
    return tf.reshape(tf.transpose(tensor, indies), (int(row_size), int(col_size)))


def vectorize(tensor):
    """
    Verctorize a tensor to a vector
    :param tensor: tf.Tensor, ndarray
    :return: vector-like tf.Tensor
    """
    return tf.reshape(tensor, -1)


def vec_to_tensor(vec, shape):
    """
    Transfer a vector to a specified shpae tensor
    :param vec:
    :param shape:
    :return:
    """
    return tf.reshape(vec, shape)


def dot(tensorA, tensorB, a_axis, b_axis):
    """

    :param tensorA:
    :param tensorB:
    :param a_axis:
    :param b_axis:
    :return:
    """
    A = t2mat(tensorA, a_axis, -1)
    B = t2mat(tensorB, b_axis, -1)
    mat_dot = tf.matmul(A, B, transpose_b=True)
    back_shape = [_ for _ in range(len(tensorA.shape)) if _ not in a_axis] + \
                 [_ for _ in range(len(tensorB.shape)) if _ not in b_axis]
    return tf.reshape(mat_dot, back_shape)

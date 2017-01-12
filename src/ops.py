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
    return tf.reshape(tensor, [-1])


def vec_to_tensor(vec, shape):
    """
    Transfer a vector to a specified shape tensor
    :param vec: a vector-like tensor
    :param shape: list, tuple
    :return: TTensor
    """
    return tf.reshape(vec, shape)


def mul(tensorA, tensorB, a_axis, b_axis):
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


def inner(tensorA, tensorB):
    """

    :param tensorA:
    :param tensorB:
    :return:
    """
    return tf.reduce_sum(vectorize(tensorA) * vectorize(tensorB))


def kron(matrices, skip_matrices_index=None, reverse=False):
    """

    :param matrices:
    :param skip_matrices_index:
    :param reverse:
    :return:
    """
    if skip_matrices_index is not None:
        matrices = [matrices[_] if isinstance(matrices[_], tf.Tensor) else tf.constant(matrices[_])
                    for _ in range(len(matrices)) if _ not in skip_matrices_index]
    else:
        matrices = [mat if isinstance(mat, tf.Tensor) else tf.constant(mat)
                    for mat in matrices]
    if reverse:
        matrices = matrices[::-1]
    start = ord('a')
    source = ','.join(chr(start + i) + chr(start + i + 1) for i in range(0, len(matrices), 2))
    row = ''.join(chr(start + i) for i in range(0, len(matrices), 2))
    col = ''.join(chr(start + i) for i in range(1, len(matrices), 2))
    operation = source + '->' + row + col
    tmp = tf.einsum(operation, *matrices)
    r_size = np.prod([int(mat.get_shape()[0]) for mat in matrices])
    c_size = np.prod([int(mat.get_shape()[1]) for mat in matrices])
    back_shape = (r_size, c_size)
    return tf.reshape(tmp, back_shape)


def khatri(matrices, skip_matrices_index=None, reverse=False):
    """

    :param matrices:
    :param skip_matrices_index:
    :param reverse:
    :return:
    """
    if skip_matrices_index is not None:
        matrices = [tf.constant(matrices[_]) if isinstance(matrices[_], np.ndarray) else matrices[_]
                    for _ in range(len(matrices)) if _ not in skip_matrices_index]
    else:
        matrices = [tf.constant(mat) if isinstance(mat, np.ndarray) else mat
                    for mat in matrices]
    if reverse:
        matrices = matrices[::-1]
    start = ord('a')
    common_dim = 'z'
    target = ''.join(chr(start + i) for i in range(len(matrices)))
    source = ','.join(i + common_dim for i in target)
    operation = source + '->' + target + common_dim
    tmp = tf.einsum(operation, *matrices)
    r_size = np.prod([int(mat.get_shape()[0]) for mat in matrices])
    back_shape = (r_size, int(matrices[0].get_shape()[1]))
    return tf.reshape(tmp, back_shape)

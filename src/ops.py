# Created by ay27 at 17/1/11
import tensorflow as tf
import numpy as np


def _gen_perm(order, mode):
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
    :param tensor: tf.Tensor
    :param mode: int, default is 0
    :return: tf.Tensor
    """
    perm = _gen_perm(tensor.get_shape().ndims, mode)
    return tf.reshape(tf.transpose(tensor, perm), (tensor.get_shape().as_list()[mode], -1))


def fold(unfolded_tensor, mode, shape):
    """
    Fold an unfolded tensor to tensor with specified shape
    :param unfolded_tensor: tf.Tensor
    :param mode: int
    :param shape: the specified shape of target tensor
    :return: tf.Tensor
    """
    perm = _gen_perm(len(shape), mode)
    shape_now = [shape[_] for _ in perm]
    back_perm = [item[0] for item in sorted(enumerate(perm), key=lambda x: x[1])]
    return tf.transpose(tf.reshape(unfolded_tensor, shape_now), back_perm)


def t2mat(tensor, r_axis, c_axis):
    """
    Transfer a tensor to a matrix by given row axis and column axis
    :param tensor: tf.Tensor
    :param r_axis: int, list
    :param c_axis: int, list
    :return: matrix-like tf.Tensor
    """
    if isinstance(r_axis, int):
        indies = [r_axis]
        row_size = tensor.get_shape()[r_axis].value
    else:
        indies = r_axis
        row_size = np.prod([tensor.get_shape()[i].value for i in r_axis])
    if c_axis == -1:
        c_axis = [_ for _ in range(tensor.get_shape().ndims) if _ not in r_axis]
    if isinstance(c_axis, int):
        indies.append(c_axis)
        col_size = tensor.get_shape()[c_axis].value
    else:
        indies = indies + c_axis
        col_size = np.prod([tensor.get_shape()[i].value for i in c_axis])
    return tf.reshape(tf.transpose(tensor, indies), (int(row_size), int(col_size)))


def vectorize(tensor):
    """
    Verctorize a tensor to a vector
    :param tensor: tf.Tensor
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
    Multiple tensor A and tensor B by the axis of a_axis and b_axis
    :param tensorA: tf.Tensor
    :param tensorB: tf.Tensor
    :param a_axis: list
    :param b_axis: list
    :return: tf.Tensor
    """
    A = t2mat(tensorA, a_axis, -1)
    B = t2mat(tensorB, b_axis, -1)
    mat_dot = tf.matmul(A, B, transpose_a=True)
    back_shape = [tensorA.get_shape()[_].value for _ in range(tensorA.get_shape().ndims) if _ not in a_axis] + \
                 [tensorB.get_shape()[_].value for _ in range(tensorB.get_shape().ndims) if _ not in b_axis]
    return tf.reshape(mat_dot, back_shape)


def inner(tensorA, tensorB):
    """
    Inner product or tensor A and tensor B. The shape of A and B must be equal.
    :param tensorA: tf.Tensor
    :param tensorB: tf.Tensor
    :return: constant-like tf.Tensor
    :raise: ValueError
    raise if the shape of A and B not equal
    """
    if tensorA.get_shape() != tensorB.get_shape():
        raise ValueError('the shape of tensor A and B must be equal')
    return tf.reduce_sum(vectorize(tensorA) * vectorize(tensorB))


def hadamard(matrices, skip_matrices_index=None, reverse=False):
    """
    Hadamard product of given matrices, which is the element product of matrix.
    :param matrices: List

    :param skip_matrices_index: List
    skip some matrices

    :param reverse: bool
    reverse the matrices order

    :return: tf.Tensor
    """
    if skip_matrices_index is not None:
        matrices = [matrices[_] for _ in range(len(matrices)) if _ not in skip_matrices_index]
    if reverse:
        matrices = matrices[::-1]
    res = tf.eye(matrices[0].get_shape()[0], matrices[0].get_shape()[1])
    for mat in matrices:
        res *= mat
    return res


def kron(matrices, skip_matrices_index=None, reverse=False):
    """
    Kronecker product of given matrices.
    :param matrices: List

    :param skip_matrices_index: List

    :param reverse: bool

    :return: tf.Tensor
    """
    if skip_matrices_index is not None:
        matrices = [matrices[_] for _ in range(len(matrices)) if _ not in skip_matrices_index]
    if reverse:
        matrices = matrices[::-1]
    start = ord('a')
    source = ','.join(chr(start + i) + chr(start + i + 1) for i in range(0, 2 * len(matrices), 2))
    row = ''.join(chr(start + i) for i in range(0, len(matrices), 2))
    col = ''.join(chr(start + i) for i in range(1, len(matrices), 2))
    operation = source + '->' + row + col
    tmp = tf.einsum(operation, *matrices)
    r_size = tf.reduce_prod([mat.get_shape()[0].value for mat in matrices])
    c_size = tf.reduce_prod([mat.get_shape()[1].value for mat in matrices])
    back_shape = (r_size, c_size)
    return tf.reshape(tmp, back_shape)


def khatri(matrices, skip_matrices_index=None, reverse=False):
    """
    Khatri-Rao product
    :param matrices: List

    :param skip_matrices_index: List

    :param reverse: bool

    :return: tf.Tensor
    """
    if skip_matrices_index is not None:
        matrices = [matrices[_] for _ in range(len(matrices)) if _ not in skip_matrices_index]
    if reverse:
        matrices = matrices[::-1]
    start = ord('a')
    common_dim = 'z'

    target = ''.join(chr(start + i) for i in range(len(matrices)))
    source = ','.join(i + common_dim for i in target)
    operation = source + '->' + target + common_dim
    tmp = tf.einsum(operation, *matrices)
    r_size = tf.reduce_prod([int(mat.get_shape()[0].value) for mat in matrices])
    back_shape = (r_size, int(matrices[0].get_shape()[1].value))
    return tf.reshape(tmp, back_shape)

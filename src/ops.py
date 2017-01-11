# Created by ay27 at 17/1/11
import tensorflow as tf
import numpy as np


def __gen_perm(order, mode):
    tmp = list(range(order - 1, -1, -1))
    tmp.remove(mode)
    perm = [mode] + tmp
    return perm


def unfold(tensor, mode=0):
    """
    Unfold tensor to a matrix, using Kolda-type
    :param tensor: ndarray
    :param mode: int, default is 0
    :return: tf.Tensor
    """
    perm = __gen_perm(len(tensor.shape), mode)
    return tf.reshape(tf.transpose(tensor, perm), (tensor.shape[mode], -1))


def fold(unfolded_tensor, mode, shape):
    """

    :param unfolded_tensor:
    :param mode:
    :param shape:
    :return:
    """
    perm = __gen_perm(len(shape), mode)
    shape_now = [shape[_] for _ in perm]
    back_perm = [item[0] for item in sorted(enumerate(perm), key=lambda x: x[1])]
    return tf.transpose(tf.reshape(unfolded_tensor, shape_now), back_perm)

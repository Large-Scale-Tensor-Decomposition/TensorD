# Created by ay27 at 17/1/11
import tensorflow as tf
import numpy as np


def unfold(tensor, mode=0):
    """
    Unfold tensor to a matrix, using Kolda-type
    :param tensor: ndarray
    :param mode: int, default is 0
    :return: tf.Tensor
    """
    tmp = list(range(len(tensor.shape) - 1, -1, -1))
    tmp.remove(mode)
    perm = [mode] + tmp
    return tf.reshape(tf.transpose(tensor, perm), (tensor.shape[mode], -1))

def fold(unfolded_tensor, mode, shape):
    """

    :param unfolded_tensor:
    :param mode:
    :param shape:
    :return:
    """

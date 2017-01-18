# Created by ay27 at 17/1/17
import tensorflow as tf
import numpy as np


def rmse(diff_tensor):
    """
    RMSE = \sqrt{ \frac{1}{c} \left \| R - U \bar{V}  \right \|_F^2}
         = \sqrt{ \frac{1}{c} \sum (R_{ij} - {U_{*i}^T}\bar{V}_{*j})^2}
    :param diff_tensor:
    :return:
    """
    return tf.sqrt(tf.reduce_sum(tf.square(diff_tensor)) / diff_tensor.get_shape().num_elements())

# Created by ay27 at 17/2/23
import tensorflow as tf


def l2(f, h):
    """
    loss = \frac{1}{2} ||f - h||^2_2
    Parameters
    ----------
    f
    h

    Returns
    -------

    """
    return 0.5 * tf.reduce_sum(tf.square(f - h))


def rmse(diff_tensor):
    """
    RMSE = \sqrt{ \frac{1}{c} \left \| R - U \bar{V}  \right \|_F^2}
         = \sqrt{ \frac{1}{c} \sum (R_{ij} - {U_{*i}^T}\bar{V}_{*j})^2}
    :param diff_tensor:
    :return:
    """
    return tf.sqrt(tf.reduce_sum(tf.square(diff_tensor)) / diff_tensor.get_shape().num_elements())

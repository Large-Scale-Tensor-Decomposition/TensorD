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

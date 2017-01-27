# Created by ay27 on 2017/1/27
import unittest
import factorizer.factorization.fact as fact
import time
import tensorflow as tf
import numpy as np
import logging

rand = np.random.rand
logger = logging.getLogger('TEST')


class MyTestCase(unittest.TestCase):
    def test_cp(self):
        # x = rand(30, 40, 50)
        x = np.arange(60).reshape(3,4,5)

        ts = time.time()

        sess = tf.Session()
        with sess.as_default():
            fact.CP_ALS(sess, tf.constant(x, dtype=tf.float64),
                        rank=3, steps=200, learning_rate=0.0004)
            # logger.info(res)

        logger.info(time.time() - ts)


if __name__ == '__main__':
    unittest.main()

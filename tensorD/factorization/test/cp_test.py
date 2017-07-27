# # Created by ay27 at 17/1/17
# import time
# import unittest
#
# import numpy as np
# import tensorflow as tf
# import logging
# from tensorD.factorization.cp import cp, fake_cp
# from numpy.random import rand
#
# assert_array_equal = np.testing.assert_array_almost_equal
#
# logger = logging.getLogger('TEST')
#
#
# class MyTestCase(unittest.TestCase):
#     def test_cp(self):
#         # x = np.arange(60000).reshape(30, 40, 50)
#         # x = rand(30, 40, 50)
#         x = np.arange(60).reshape(3,4,5)
#
#         ts = time.time()
#
#         sess = tf.Session()
#         with sess.as_default():
#             res, _ = cp(sess, tf.constant(x, dtype=tf.float64), 2, steps=10, get_rmse=True, verbose=True, ignore_tol=False)
#             logger.info(res)
#
#         logger.info(time.time() - ts)
#         #
#         # ts = time.time()
#         # num_core = 12
#         # config = tf.ConfigProto(inter_op_parallelism_threads=num_core,
#         #                         intra_op_parallelism_threads=num_core)
#         # with tf.Session(config=config) as sess:
#         #     res, _ = cp(sess, tf.constant(x, dtype=tf.float64), 20, steps=20, get_rmse=True)
#         #     logger.info(res)
#         # logger.info(time.time() - ts)
#
#     def test_fake_cp(self):
#         # x = rand(30, 40, 50)
#         x = np.arange(60).reshape(3,4,5)
#
#         ts = time.time()
#
#         sess = tf.Session()
#         with sess.as_default():
#             fake_cp(sess, tf.constant(x, dtype=tf.float64), 2, steps=100)
#             # logger.info(res)
#
#         logger.info(time.time() - ts)
#
#
# if __name__ == '__main__':
#     unittest.main()

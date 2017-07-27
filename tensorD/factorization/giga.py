# # Created by ay27 at 17/4/5
# import tensorflow as tf
# import numpy as np
#
# import tensorD.loss as ltp
# import tensorD.base.ops as ops
# from tensorD.base.type import KTensor
#
#
# # Define parameters
# FLAGS = tf.app.flags.FLAGS
#
# # For distributed
# tf.app.flags.DEFINE_string("ps_hosts", "",
#                            "Comma-separated list of hostname:port pairs")
# tf.app.flags.DEFINE_string("worker_hosts", "",
#                            "Comma-separated list of hostname:port pairs")
# tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
# tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
# tf.app.flags.DEFINE_integer("task_cnt", 2, "count of tasks")
#
#
# def _skip(matrices, skip_matrices_index):
#     if skip_matrices_index is not None:
#         if isinstance(skip_matrices_index, int):
#             skip_matrices_index = [skip_matrices_index]
#         return [matrices[_] for _ in range(len(matrices)) if _ not in skip_matrices_index]
#     return matrices
#
#
# def khatri(matrices, skip_matrices_index=None, reverse=False):
#     matrices = _skip(matrices, skip_matrices_index)
#     if reverse:
#         matrices = matrices[::-1]
#     start = ord('a')
#     common_dim = 'z'
#
#     target = ''.join(chr(start + i) for i in range(len(matrices)))
#     source = ','.join(i + common_dim for i in target)
#     operation = source + '->' + target + common_dim
#     tmp = tf.einsum(operation, *matrices)
#     r_size = tf.reduce_prod([int(mat.get_shape()[0].value) for mat in matrices])
#     back_shape = (r_size, int(matrices[0].get_shape()[1].value))
#     return tf.reshape(tmp, back_shape)
#
#
# def extract(U):
#     tmp = khatri(U)
#     lambdas = tf.ones((U[0].get_shape()[1].value, 1), dtype=tf.float32)
#     back_shape = [u.get_shape()[0].value for u in U]
#     return tf.reshape(tf.matmul(tmp, lambdas), back_shape)
#
#
#
# def main(_):
#     ps_hosts = FLAGS.ps_hosts.split(",")
#     worker_hosts = FLAGS.worker_hosts.split(",")
#     cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
#     server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
#
#     I = 30
#     J = 40
#     K = 50
#     STEP = 100
#     tensor = np.random.rand(I, J, K)
#     tensor = tf.constant(tensor, dtype=tf.float64)
#     R = 20
#
#     order = 3
#
#     job_name = FLAGS.job_name
#     task_index = FLAGS.task_index
#     task_cnt = FLAGS.task_cnt
#
#     if job_name == 'ps':
#         server.join()
#     elif job_name == 'worker':
#         for worker in range(task_cnt):
#             with tf.device("/job:worker/task:%d" % worker):
#                 X1 = ops.unfold(tensor, 0)
#                 X2 = ops.unfold(tensor, 1)
#                 X3 = ops.unfold(tensor, 2)
#                 A = tf.get_variable("A%d" % worker, (I, R), dtype=tensor.dtype)
#                 B = tf.get_variable("B%d" % worker, (J, R), dtype=tensor.dtype)
#                 C = tf.get_variable("C%d" % worker, (K, R), dtype=tensor.dtype)
#
#
#
#
#
# if __name__ == "__main__":
#     tf.app.run()

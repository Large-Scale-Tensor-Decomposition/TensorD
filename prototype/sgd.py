# Created by ay27 at 17/2/9
import tensorflow as tf
import numpy as np
import time

"""
run:
python train.py --ps_hosts=172.17.0.3:2222 --worker_hosts=172.17.0.4:2223 --job_name=worker
python train.py --ps_hosts=172.17.0.3:2222 --worker_hosts=172.17.0.4:2223 --job_name=ps
"""

# Define parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('learning_rate', 0.003, 'Initial learning rate.')
tf.app.flags.DEFINE_integer('steps_to_validate', 100,
                            'Steps to validate and print loss')

# For distributed
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

# Hyperparameters
learning_rate = FLAGS.learning_rate
steps_to_validate = FLAGS.steps_to_validate


def _skip(matrices, skip_matrices_index):
    if skip_matrices_index is not None:
        if isinstance(skip_matrices_index, int):
            skip_matrices_index = [skip_matrices_index]
        return [matrices[_] for _ in range(len(matrices)) if _ not in skip_matrices_index]
    return matrices


def khatri(matrices, skip_matrices_index=None, reverse=False):
    matrices = _skip(matrices, skip_matrices_index)
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


def extract(U):
    tmp = khatri(U)
    lambdas = tf.ones((U[0].get_shape()[1].value, 1), dtype=tf.float32)
    back_shape = [u.get_shape()[0].value for u in U]
    return tf.reshape(tf.matmul(tmp, lambdas), back_shape)


def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    I = 10
    J = 20
    K = 30
    tensor = np.random.rand(I, J, K)
    R = 20

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):
            global_step = tf.Variable(0, name='global_step', trainable=False)

            X = tf.placeholder(dtype=tf.float32)

            A = tf.get_variable("A", [I, R], tf.float32, initializer=tf.random_normal_initializer())
            B = tf.get_variable("B", [J, R], tf.float32, initializer=tf.uniform_unit_scaling_initializer())
            C = tf.get_variable("C", [K, R], tf.float32, initializer=tf.uniform_unit_scaling_initializer())

            pred = extract([A, B, C])

            loss_value = tf.sqrt(tf.reduce_sum(tf.square(X - pred)))

            train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_value, global_step=global_step)

            saver = tf.train.Saver()
            tf.summary.histogram('loss', loss_value)
            summary_op = tf.summary.merge_all()
            init_op = tf.global_variables_initializer()

            print(FLAGS.task_index)

        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                 init_op=init_op,
                                 summary_op=summary_op,
                                 saver=saver,
                                 global_step=global_step,
                                 save_model_secs=60)
        with sv.managed_session(server.target) as sess:
            if FLAGS.task_index == 0:
                writer = tf.summary.FileWriter('/code/log', sess.graph)
            old_ts = time.time()
            step = 0
            print(FLAGS.task_index)
            while step < 5000:
                # print(FLAGS.task_index)
                _, loss_v, step = sess.run([train_op, loss_value, global_step],
                                           feed_dict={X: tensor})
                if step % steps_to_validate == 0:
                    print('step %d, loss=%f' % (step, loss_v))
                    if FLAGS.task_index == 0:
                        sum_str = sess.run(summary_op, feed_dict={X: tensor})
                        writer.add_summary(sum_str)

            print('task %d, cost time : %f' % (FLAGS.task_index, time.time() - old_ts))
            print('cost time : %f' % (time.time() - old_ts))

        sv.stop()


if __name__ == "__main__":
    tf.app.run()

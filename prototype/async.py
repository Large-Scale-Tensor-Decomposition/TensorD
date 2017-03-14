# Created by ay27 at 17/2/9
import pickle

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
tf.app.flags.DEFINE_float('learning_rate', 0.0005, 'Initial learning rate.')
tf.app.flags.DEFINE_integer('steps_to_validate', 100,
                            'Steps to validate and print loss')

# For distributed
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("task_cnt", 3, "Total count of workers")

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


def average_grads(grads):
    average_grads = []
    for grad_and_vars in zip(*grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    I = 3
    J = 4
    K = 5
    STEP = 10000
    # tensor = np.random.rand(I, J, K)
    tensor = np.arange(I*J*K).reshape(I,J,K)
    R = 2

    if FLAGS.job_name == 'ps':
        server.join()
    else:
        print("before construct graph %d" % FLAGS.task_index)

        with tf.device(tf.train.replica_device_setter(cluster=cluster)):
            global_step = tf.Variable(0, name='global_step', trainable=False)

            X = tf.placeholder(dtype=tf.float32)

            A = tf.get_variable("A", [I, R], tf.float32, initializer=tf.random_normal_initializer())
            B = tf.get_variable("B", [J, R], tf.float32, initializer=tf.uniform_unit_scaling_initializer())
            C = tf.get_variable("C", [K, R], tf.float32, initializer=tf.uniform_unit_scaling_initializer())

        opt = tf.train.GradientDescentOptimizer(learning_rate)
        grads = []

        print("before construct model %d" % FLAGS.task_index)

        for ii in range(int(FLAGS.task_cnt)):
            with tf.device("/job:worker/task:%d" % ii):
                with tf.name_scope("worker%d" % ii):
                    pred = extract([A, B, C])

                    loss = tf.sqrt(tf.reduce_sum(tf.square(X - pred)))
                    tf.summary.histogram("loss", loss)

                    grad = opt.compute_gradients(loss)

                    grads.append(grad)

        aver_grad = average_grads(grads)

        apply_gradient_op = opt.apply_gradients(aver_grad, global_step=global_step)
        variable_averages = tf.train.ExponentialMovingAverage(0.01, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        g_loss = tf.sqrt(tf.reduce_sum(tf.square(X - extract([A, B, C]))))

        train_op = tf.group(apply_gradient_op, variables_averages_op)

        saver = tf.train.Saver()
        tf.summary.histogram('loss', loss)
        summary_op = tf.summary.merge_all()
        init_op = tf.global_variables_initializer()

        print("after model %d" % FLAGS.task_index)

        summary_writer = tf.summary.FileWriter('/code/tmp/', graph=tf.get_default_graph())

        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                 init_op=init_op,
                                 summary_op=summary_op,
                                 saver=saver,
                                 global_step=global_step,
                                 save_model_secs=60)
        with sv.managed_session(server.target) as sess:
            # with tf.Session(server.target) as sess:
            # if FLAGS.task_index == 0:
            #     summary = []
            summary = []
            step = 0
            print('start train, worker %d, job as %s' % (FLAGS.task_index, FLAGS.job_name))
            while step < STEP:
                # print("step %d, worker %d" % (step, FLAGS.task_index))
                _, step, loss_v, loss_g = sess.run([train_op, global_step, loss, g_loss], feed_dict={X: tensor})
                sum_str = sess.run(summary_op, feed_dict={X: tensor})
                summary_writer.add_summary(sum_str)
                summary.append([time.time(), loss_g, loss_v])
                if step % 50 == 0:
                    print('loss = %f, global loss = %f' % (loss_v, loss_g))
                    # if FLAGS.task_index == 0:
                    # summary.append([time.time() - old_ts, loss_v])
                    # if step % steps_to_validate == 0:
                    #     print('step %d, loss=%f' % (step, loss_v))

            # print('cost time : %f' % (time.time() - old_ts))
            # if FLAGS.task_index == 0:
            with open('/code/log/w_%d' % FLAGS.task_index, 'wb') as file:
                pickle.dump(summary, file)
            with open('/code/log/wa_%d' % FLAGS.task_index, 'wb') as file:
                pickle.dump(sess.run(A), file)
            with open('/code/log/wb_%d' % FLAGS.task_index, 'wb') as file:
                pickle.dump(sess.run(B), file)
            with open('/code/log/wc_%d' % FLAGS.task_index, 'wb') as file:
                pickle.dump(sess.run(C), file)

            sv.stop()


if __name__ == "__main__":
    tf.app.run()

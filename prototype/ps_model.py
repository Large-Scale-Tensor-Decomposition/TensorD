# Created by ay27 at 17/3/22
import tensorflow as tf
import numpy as np

# Define parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('learning_rate', 0.005, 'Initial learning rate.')
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
ps_num = len(FLAGS.ps_hosts.split(','))
worker_num = len(FLAGS.worker_hosts.split(','))


def log(msg):
    print('%s task %d : %s' % (FLAGS.job_name, FLAGS.task_index, msg))


def push(server_op_name, worker_op_name, worker_id):
    op1 = None
    op2 = None
    with tf.device('job:ps/task0'):
        op1 = tf.get_variable(server_op_name)
    with tf.device('job:worker/task%d' % worker_id):
        op2 = tf.get_variable(worker_op_name)

    return op1.assign_add(op2)


def pull(server_op_name, worker_op_name, worker_id):
    op1 = None
    op2 = None
    with tf.device('job:ps/task0'):
        op1 = tf.get_variable(server_op_name)
    with tf.device('job:worker/task%d' % worker_id):
        op2 = tf.get_variable(worker_op_name)

    return op2.assign(op1)


def create_graph():
    for i in range(worker_num):
        x = tf.placeholder(tf.float64)
        y = tf.placeholder(tf.float64)

        # w = tf.get_variable(name='w%d' % i, shape=[1], dtype=tf.float64)
        # b = tf.get_variable(name='b%d' % i, shape=[1], dtype=tf.float64)
        w = pull('w', 'w%d' % i, i)
        b = pull('b', 'b%d' % i, i)

        pred = tf.multiply(w, x) + b

        loss = tf.losses.mean_squared_error(y, pred)

        opt = tf.train.GradientDescentOptimizer(learning_rate)

        grads = opt.compute_gradients(loss, var_list=[w, b])
        apply_op = opt.apply_gradients(grads)

        push('w', w.name, i)
        push('b', b.name, i)

    return loss, apply_op


def run_ps():
    log('run ps')
    create_graph()


def run_worker():
    log('run worker')


def launch_ps():
    pass


def launch_workers():
    pass


def main(_):
    if FLAGS.job_name == "ps":
        run_ps()
    elif FLAGS.job_name == "worker":
        run_worker()


if __name__ == '__main__':
    tf.app.run()

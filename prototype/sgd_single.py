# Created by ay27 at 17/2/9
import tensorflow as tf
import numpy as np
import time
import pickle


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


def main():
    I = 10
    J = 20
    K = 30
    STEP = 1000
    tensor = np.random.rand(I, J, K)
    R = 20

    global_step = tf.Variable(0, name='global_step', trainable=False)

    X = tf.placeholder(dtype=tf.float32)

    A = tf.get_variable("A", [I, R], tf.float32, initializer=tf.random_normal_initializer())
    B = tf.get_variable("B", [J, R], tf.float32, initializer=tf.uniform_unit_scaling_initializer())
    C = tf.get_variable("C", [K, R], tf.float32, initializer=tf.uniform_unit_scaling_initializer())

    pred = extract([A, B, C])

    loss_value = tf.sqrt(tf.reduce_sum(tf.square(X - pred)))

    gd_op = tf.train.GradientDescentOptimizer(0.003)
    grad = gd_op.compute_gradients(loss_value)
    train_op = gd_op.apply_gradients(grad)

    tf.summary.histogram('loss', loss_value)
    sum_op = tf.summary.merge_all()
    sum_writer = tf.summary.FileWriter('/tmp/sgd_s', tf.get_default_graph())

    init_op = tf.global_variables_initializer()

    summary = []

    with tf.Session() as sess:
        sess.run(init_op)
        old_ts = time.time()
        step = 0
        while step < STEP:
            _, loss_v, step = sess.run([train_op, loss_value, global_step],
                                       feed_dict={X: tensor})
            summary.append([time.time() - old_ts, loss_v])
            sum_str = sess.run(sum_op, feed_dict={X: tensor})
            sum_writer.add_summary(sum_str)


        print('cost time : %f' % (time.time() - old_ts))

    # with open('/home/ay27/tf/prototype/log/s_%d%d%d' % (I, J, K), 'wb') as file:
    #     pickle.dump(summary, file)


if __name__ == "__main__":
    main()

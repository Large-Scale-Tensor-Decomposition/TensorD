# Created by ay27 at 17/1/12
import numpy as np
import tensorflow as tf

import factorizer.base.ops as ops

rand = np.random.rand


def CP_ALS(sess, tensor, rank, steps=100, learning_rate=0.002):
    # X1 = ops.unfold(tensor, 0)
    # X2 = ops.unfold(tensor, 1)
    # X3 = ops.unfold(tensor, 2)
    #
    # shape = tf.shape(tensor).eval()
    #
    # A = tf.Variable(rand(shape[0], rank))
    # B = tf.Variable(rand(shape[1], rank))
    # C = tf.Variable(rand(shape[2], rank))
    #
    # loss1 = tf.reduce_sum(tf.square(X1 - tf.matmul(A, ops.khatri([B, C]), transpose_b=True)))
    # loss2 = tf.reduce_sum(tf.square(X2 - tf.matmul(B, ops.khatri([C, A]), transpose_b=True)))
    # loss3 = tf.reduce_sum(tf.square(X3 - tf.matmul(C, ops.khatri([B, A]), transpose_b=True)))
    #
    # optimizer1 = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss1, var_list=[A])
    # optimizer2 = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss2, var_list=[B])
    # optimizer3 = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss3, var_list=[C])
    #
    # init = tf.global_variables_initializer()
    #
    # sess.run(init)
    # for step in range(steps):
    #     _ = sess.run(optimizer1)
    #     _ = sess.run(optimizer2)
    #     _ = sess.run(optimizer3)
    #     loss = sess.run(loss3)
    #     print('step %d, loss = %f' % (step, loss))
    # print(sess.run(X3 - tf.matmul(C, ops.khatri([B, A]), transpose_b=True)))

    X1 = ops.unfold(tensor, 0)
    X2 = ops.unfold(tensor, 1)
    X3 = ops.unfold(tensor, 2)

    shape = tf.shape(tensor).eval()
    A = tf.placeholder(tf.float64, (shape[0], rank))
    B = tf.placeholder(tf.float64, (shape[1], rank))
    C = tf.placeholder(tf.float64, (shape[2], rank))

    V1 = ops.hadamard([tf.matmul(B, B, transpose_a=True), tf.matmul(C, C, transpose_a=True)])
    V2 = ops.hadamard([tf.matmul(A, A, transpose_a=True), tf.matmul(C, C, transpose_a=True)])
    V3 = ops.hadamard([tf.matmul(A, A, transpose_a=True), tf.matmul(B, B, transpose_a=True)])

    A1 = tf.matmul(X1, tf.matmul(ops.khatri([C, B]), tf.matrix_inverse(V1)))
    A2 = tf.matmul(X2, tf.matmul(ops.khatri([C, A]), tf.matrix_inverse(V2)))
    A3 = tf.matmul(X3, tf.matmul(ops.khatri([B, A]), tf.matrix_inverse(V3)))

    XX = tf.matmul(A1, ops.khatri([C, B]), transpose_b=True)
    loss = tf.reduce_sum(tf.square(XX - X1))

    init = tf.global_variables_initializer()

    sess.run(init)
    a = np.random.rand(shape[0], rank)
    b = np.random.rand(shape[1], rank)
    c = np.random.rand(shape[2], rank)
    for step in range(steps):
        v1, v2, v3 = sess.run([V1, V2, V3], feed_dict={A: a, B: b, C: c})
        a, b, c = sess.run([A1, A2, A3], feed_dict={A: a, B: b, C: c, V1: v1, V2: v2, V3: v3})
        res = sess.run(loss, feed_dict={A1: a, B: b, C: c})
        print('step %d, loss = %f' % (step, res))

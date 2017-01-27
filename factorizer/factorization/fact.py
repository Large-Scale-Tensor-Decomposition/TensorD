# Created by ay27 at 17/1/12
import tensorflow as tf
import numpy as np
import factorizer.base.ops as ops

rand = np.random.rand


def CP_ALS(sess, tensor, rank, steps=100, learning_rate=0.002):
    X1 = ops.unfold(tensor, 0)
    X2 = ops.unfold(tensor, 1)
    X3 = ops.unfold(tensor, 2)

    shape = tensor.get_shape().as_list()

    vA = tf.Variable(rand(shape[0], rank))
    vB = tf.Variable(rand(shape[1], rank))
    vC = tf.Variable(rand(shape[2], rank))
    # A = tf.placeholder(tf.float64, shape=(shape[0], rank))
    # B = tf.placeholder(tf.float64, shape=(shape[1], rank))
    # C = tf.placeholder(tf.float64, shape=(shape[2], rank))

    loss1 = tf.reduce_sum(tf.square(X1 - tf.matmul(vA, ops.khatri([vB, vC]), transpose_b=True)))
    # loss2 = tf.reduce_sum(tf.square(X2 - tf.matmul(vB, ops.khatri([C, A]), transpose_b=True)))
    # loss3 = tf.reduce_sum(tf.square(X3 - tf.matmul(vC, ops.khatri([B, A]), transpose_b=True)))

    optimizer1 = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss1)
    # optimizer2 = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss2, var_list=[vB])
    # optimizer3 = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss3, var_list=[vC])

    # loss = tf.reduce_sum(tf.square(X3 - tf.matmul(C, ops.khatri([B, A]), transpose_b=True)))

    init = tf.global_variables_initializer()

    sess.run(init)

    # a = rand(shape[0], rank)
    # b = rand(shape[1], rank)
    # c = rand(shape[2], rank)

    for step in range(steps):
        _ = sess.run([optimizer1])
        # a, b, c = sess.run([vA, vB, vC])
        ll = sess.run(loss1)
        print('step %d, loss = %f' % (step, ll))
    # print(sess.run(X3 - tf.matmul(C, ops.khatri([B, A]), transpose_b=True)))

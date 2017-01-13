# Created by ay27 at 17/1/12
import tensorflow as tf
import numpy as np
import src.ops as ops

rand = np.random.rand


def CP(sess, tensor, rank, steps=100, learning_rate=0.002):
    X1 = ops.unfold(tensor, 0)
    X2 = ops.unfold(tensor, 1)
    X3 = ops.unfold(tensor, 2)

    shape = tf.shape(tensor).eval()

    A = tf.Variable(rand(shape[0], rank))
    B = tf.Variable(rand(shape[1], rank))
    C = tf.Variable(rand(shape[2], rank))

    loss1 = tf.reduce_sum(tf.square(X1 - tf.matmul(A, ops.khatri([B, C]), transpose_b=True)))
    loss2 = tf.reduce_sum(tf.square(X2 - tf.matmul(B, ops.khatri([C, A]), transpose_b=True)))
    loss3 = tf.reduce_sum(tf.square(X3 - tf.matmul(C, ops.khatri([B, A]), transpose_b=True)))

    optimizer1 = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss1, var_list=[A])
    optimizer2 = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss2, var_list=[B])
    optimizer3 = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss3, var_list=[C])

    init = tf.global_variables_initializer()

    sess.run(init)
    for step in range(steps):
        _ = sess.run(optimizer1)
        _ = sess.run(optimizer2)
        _ = sess.run(optimizer3)
        loss = sess.run(loss3)
        print('step %d, loss = %f' % (step, loss))
    print(sess.run(X3 - tf.matmul(C, ops.khatri([B, A]), transpose_b=True)))

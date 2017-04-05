# Created by ay27 at 17/4/5
import tensorflow as tf
import numpy as np

import tensorD.loss as ltp
import tensorD.base.ops as ops
from tensorD.base.type import KTensor


def gigatensor(sess, tensor):
    shape = tensor.get_shape().as_list()
    I = shape[0]
    J = shape[1]
    K = shape[2]
    R = 20
    STEPS = 1000

    A = tf.get_variable("A", shape=(I, R), dtype=tf.float64, initializer=tf.random_uniform_initializer())
    B = tf.get_variable("B", shape=(J, R), dtype=tf.float64, initializer=tf.random_uniform_initializer())
    C = tf.get_variable("C", shape=(K, R), dtype=tf.float64, initializer=tf.random_uniform_initializer())

    X1 = ops.unfold(tensor, 0)
    X2 = ops.unfold(tensor, 1)
    X3 = ops.unfold(tensor, 2)

    AtA = tf.matmul(A, A, transpose_a=True)
    BtB = tf.matmul(B, B, transpose_a=True)
    CtC = tf.matmul(C, C, transpose_a=True)

    kCB = ops.khatri([C, B])
    kCA = ops.khatri([C, A])
    kBA = ops.khatri([B, A])

    epsilon = tf.diag(tf.ones(R, dtype=tf.float64))

    t1 = tf.matmul(X1, kCB)
    t2 = tf.add(ops.hadamard([CtC, BtB]), epsilon)
    t3 = tf.matrix_inverse(t2)

    updateA = tf.matmul(t1, t3)
    updateB = tf.matmul(tf.matmul(X2, kCA), tf.matrix_inverse(tf.add(ops.hadamard([CtC, AtA]), epsilon)))
    updateC = tf.matmul(tf.matmul(X3, kBA), tf.matrix_inverse(tf.add(ops.hadamard([BtB, AtA]), epsilon)))

    # updateA = tf.transpose(tf.matrix_solve(tf.transpose(ops.hadamard([CtC, BtB])), tf.transpose(tf.matmul(X1, kCB))))
    # updateB = tf.transpose(tf.matrix_solve(tf.transpose(ops.hadamard([CtC, AtA])), tf.transpose(tf.matmul(X2, kCA))))
    # updateC = tf.transpose(tf.matrix_solve(tf.transpose(ops.hadamard([BtB, AtA])), tf.transpose(tf.matmul(X3, kBA))))

    assignA = tf.assign(A, updateA)
    assignB = tf.assign(B, updateB)
    assignC = tf.assign(C, updateC)

    with sess.graph.control_dependencies([assignA, assignB, assignC]):
        loss = ltp.rmse(tensor - KTensor([A, B, C]).extract())

    train_op = tf.group(assignA, assignB, assignC)

    tf.summary.histogram('loss', loss)

    init_op = tf.global_variables_initializer()
    summary_op = tf.summary.merge_all()
    sum_writer = tf.summary.FileWriter("/tmp/giga", sess.graph)

    # start training
    sess.run(init_op)
    for step in range(STEPS):
        _, loss_v = sess.run([train_op, loss])
        print("step %d : loss = %f" % (step, loss_v))

        sum_str = sess.run(summary_op)
        sum_writer.add_summary(sum_str)

    print("finish")

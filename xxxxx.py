#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/7/10 PM5:37
# @Author  : Shiloh Leung
# @Site    : 
# @File    : xxxxx.py
# @Software: PyCharm Community Edition
import numpy as np
import tensorflow as tf
from base.type import KTensor
import loss as loss
from numpy.random import rand
from factorization.env import Environment
from factorization.cp import CP_ALS
from dataproc.provider import Provider
from factorization.tucker import *


#h = tf.constant([[0,1,2],[3,4,5],[6,7,8]],dtype=tf.float32)
#f = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=tf.float32)
#l2_op = loss.l2(h,f)
#sess = tf.Session()
#print(sess.run(l2_op))

#A = tf.constant(np.random.rand(3,4))
#B = tf.constant(np.random.rand(3,4))
#rmse_op = loss.rmse(A, B)
#print(sess.run(rmse_op))
#A = tf.constant([[2.3, 4.3, 0],[2.3, 4.3, 0],[2.3, 4.3, 0]],dtype=tf.float64)
#B = tf.constant(np.random.rand(3,3),dtype=tf.float64)
#print(sess.run(B))
#cast_A = tf.cast(tf.not_equal(A, 0), B.dtype)
#print(sess.run(cast_A))
#_B = B * tf.cast(tf.not_equal(A, 0), B.dtype)
#print(sess.run(loss.rmse_ignore_zero(A,B)))
#def pre(*keys):
#    array = np.array([[1,2,3],[4,5,6]])
#    print(array.item(keys))



def ex2():
    state = tf.Variable(0, name = "counter")
    one = tf.constant(1)
    new_value = tf.add(state, one)
    update = tf.assign(state, new_value)
    init_op = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init_op)
        print("打印state初始值")
        print(sess.run(state))
        for _ in range(3):
            print("运行update")
            sess.run(update)
            print("打印state")
            print(sess.run(state))

def ex3():
    v = tf.Variable(0)
    new_v = v.assign(10)
    output = v + 5
    with tf.Session() as sess:
        sess.run(v.initializer)
        output_result, new_v_result = sess.run([output, new_v.op])
        print("output:")
        print(output_result)
        print("new_v")
        print(new_v_result)


def build_single_model():
    input_data = tf.constant(rand(3,4,5,3),dtype=tf.float64)
    rank = 10
    shape = input_data.get_shape().as_list()
    order = len(shape)
    A = [tf.Variable(rand(shape[ii], 10), name='A-%d' % ii) for ii in range(order)]
    mats = [ops.unfold(input_data, mode) for mode in range(order)]

    assign_op = [None for _ in range(order)]
    for mode in range(order):
        AtA = [tf.matmul(A[ii], A[ii], transpose_a=True, name='AtA-%d-%d' % (mode, ii)) for ii in range(order)]
        V = ops.hadamard(AtA, skip_matrices_index=mode)
        XA = tf.matmul(mats[mode], ops.khatri(A, mode, True), name='XA-%d' % mode)

        # TODO : does it correct to assign A[mode] with the **assign op** ?
        # But it's strangle to raise not invertible when remove the A[mode] assignment.
        assign_op[mode] = A[mode].assign(
            tf.transpose(tf.matrix_solve(tf.transpose(V), tf.transpose(XA)), name='TTT-%d' % mode))
    print("build_single_mode ends========")


def my_hosvd_test():
    data_provider = Provider()
    data_provider.full_tensor = lambda: tf.constant(rand(3, 4, 2), dtype=tf.float64)
    env = Environment(data_provider, summary_path='/tmp/tensord')
    hosvd = HOSVD(env)
    args = HOSVD.HOSVD_Args(ranks=[3,3,3])
    hosvd.build_model(args)
    hosvd.train()
    print("\nfactor matrices:")
    factors = hosvd.factors()

    for matrix in factors:
        print(matrix)
    print("\ncore matrix:")
    core = hosvd.core()
    print(core)
    P = TTensor(core, factors)
    print("\nfull tensor:")
    with tf.Session() as sess:
        seem_full = sess.run(P.extract())
    print(seem_full)
    print("\nreal full:")
    print(hosvd.full())

    print("\nsubtraction:")
    print(seem_full - hosvd.full())


def my_cp_test():
    data_provider = Provider()
    A = rand(10, 6)
    B = rand(6, 6)
    C = rand(15, 6)
    X = KTensor([A,B,C])

    data_provider.full_tensor = lambda: tf.constant(X_, dtype=tf.float64)
    env = Environment(data_provider, summary_path='/tmp/tensord')
    cp = CP_ALS(env)
    args = CP_ALS.CP_Args(rank=3, validation_internal=1000)
    cp.build_model(args)
    cp.train(10000)
    #print("\nfactor matrices:")
    #factors = cp.factors()
    #for matrix in factors:
    #    print(matrix)
    #P = KTensor(factors)
    #with tf.Session() as sess:
    #    seem_full = sess.run(P.extract())
    #print("\nseem full tensor:")
    #print(seem_full)
    #print("\nreal full:")
    #print(cp.full())

    #print("\nsubtraction:")
    #print(seem_full - cp.full())


def my_HOOI_test():
    data_provider = Provider()
    g = rand(20, 20, 20)
    A = rand(30, 20)
    B = rand(20, 20)
    C = rand(40, 20)
    X = TTensor(g, [A, B, C])
    X_ = tf.Session().run(X.extract())
    file = open('full_tensor_out.txt','w')
    print(X_,file=file)
    file.close()
    data_provider.full_tensor = lambda: tf.constant(X_, dtype=tf.float64)
    env = Environment(data_provider, summary_path='/tmp/tensord')
    hooi = HOOI(env)
    args = HOOI.HOOI_Args(ranks=[10,10,10], validation_internal=1000)
    hooi.build_model(args)
    hooi.train(10000)
    print("\nfactor matrices:")
    factors = hooi.factors()
    for matrix in factors:
        print(matrix)
    print("\ncore matrix:")
    core = hooi.core()
    print(core)
    P = TTensor(core, factors)
    print("\nfull tensor:")
    with tf.Session() as sess:
        seem_full = sess.run(P.extract())
    print(seem_full)
    print("\nreal full:")
    print(hooi.full())

    print("\nsubtraction:")
    print(seem_full - hooi.full())


my_cp_test()






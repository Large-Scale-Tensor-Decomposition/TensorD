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
from factorization import cp_xxx
from factorization.cp import CP_ALS
from dataproc.provider import Provider
from factorization.tucker import *
from DataBag import *





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


def my_cp_test(steps, print_info=False):
    data_provider = Provider()
    X_ = gen_test_tensor([3,4,5], 3)
    data_provider.full_tensor = lambda: tf.constant(X_, dtype=tf.float64)
    env = Environment(data_provider, summary_path='/tmp/tensord')
    cp = CP_ALS(env)
    args = CP_ALS.CP_Args(rank=2, validation_internal=1)
    cp.build_model(args)
    cp.train(steps)
    if print_info==True:
        print(cp.lambdas)
        print("\nfactor matrices:")
        for matrix in cp.factors:
            print(matrix)
        P = KTensor(cp.factors)
        with tf.Session() as sess:
            seem_full = sess.run(P.extract())
        print("\nseem full tensor:")
        print(seem_full)
        print("\nreal full:")
        print(X_)
        print("\nsubtraction:")
        print(seem_full - X_)


def cp_xxx_test(print_info=False):
    data_provider = Provider()
    X_ = gen_test_tensor([3,4,5], 3)
    data_provider.full_tensor = lambda: tf.constant(X_, dtype=tf.float64)
    env = Environment(data_provider, summary_path='/tmp/tensord')
    cp = cp_xxx.CP_ALS(env)
    args = cp_xxx.CP_ALS.CP_Args(rank=2, validation_internal=1)
    cp.build_model(args)
    cp.train(150)
    if print_info==True:
        print(cp.lambdas)
        print("\nfactor matrices:")
        for matrix in cp.factors:
            print(matrix)
        P = KTensor(cp.factors)
        with tf.Session() as sess:
            seem_full = sess.run(P.extract())
        print("\nseem full tensor:")
        print(seem_full)
        print("\nreal full:")
        print(X_)
        print("\nsubtraction:")
        print(seem_full - X_)




def my_HOOI_test(print_info=False):
    data_provider = Provider()
    g = rand(20, 20, 20)
    A = rand(30, 20)
    B = rand(20, 20)
    C = rand(40, 20)
    X = TTensor(g, [A, B, C])
    X_ = tf.Session().run(X.extract())
    data_provider.full_tensor = lambda: tf.constant(X_, dtype=tf.float64)
    env = Environment(data_provider, summary_path='/tmp/tensord')
    hooi = HOOI(env)
    args = HOOI.HOOI_Args(ranks=[10,10,10], validation_internal=10)
    hooi.build_model(args)
    hooi.train(10)
    if print_info == True:
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

def test1():
    order = 5
    init_op = tf.global_variables_initializer()
    A = [tf.Variable(rand(3, 6), dtype=tf.float64) for _ in range(order)]
    A_0 = tf.Variable(rand(2, 4), dtype=tf.float64)
    init_op = tf.global_variables_initializer()
    assign_op = [None for _ in range(order)]
    for mode in range(order):
        assign_op[mode] = A[mode] = A[mode].assign(tf.multiply(rand(3, 6), np.array([0, 0, 0, 0, 0, 0])))
    train_op = tf.group(*A, A_0)
    with tf.Session() as sess:
        sess.run(init_op)
        A_v, A_0_v = sess.run([A,A_0])
        ass_v = sess.run(assign_op)
        train_v = sess.run(train_op)
    print(A_v,end='\n\n')
    print(A_0_v,end='\n\n')
    print(train_v,end='\n\n')
    print(ass_v,end='\n\n')




def test_rand_list():
    print("first try")
    init_mat = rand_list([3,4,5],2)
    for mode in range(3):
        print(init_mat[mode])

    print("second try")
    init_mat2 = rand_list([2, 3, 4], 2)
    for mode in range(3):
        print(init_mat2[mode])

def test_gen_core(R):
    max_prime = 10000
    number_list = primesfrom3to(max_prime)
    core = gen_core(R, number_list)
    for i in range(R):
        print(core[:,:,i])

def test_genABC(I_list, R):
    max_prime = 10000
    number_list = primesfrom3to(max_prime)
    U = gen_ABC(I_list, R, number_list)
    for matrix in U:
        print(matrix)



def test_gen_test_tensor(I_list, R):
    full_tensor = gen_test_tensor(I_list, R)
    for i in range(I_list[-1]):
        print(full_tensor[:,:,i])



def test_assign_op():
    order = 4
    X = [tf.Variable(np.ones((ii,5))) for ii in range(order)]
    assign_op1 = [None for _ in range(order)]
    assign_op2 = [None for _ in range(order)]
    # iter 1
    for mode in range(order):
        assign_op1[mode] = X[mode].assign(X[mode]*2)
    train_op1 = tf.group(*assign_op1)

    for mode in range(order):
        assign_op2[mode] = X[mode].assign(X[mode]*3)

    init_op = tf.global_variables_initializer()
    steps = 5
    sess = tf.Session()
    sess.run(init_op)
    print("initialize:")
    res1 = sess.run(X)
    print(res1)

    print("==assign 1==")
    sess.run(train_op1)
    res1 = sess.run(assign_op1)
    print(res1)
    print("==run X now==")
    print(sess.run(X))

    #print("==assign 1-2==")
    #res1_2 = sess.run(train_op1)
    #print(res1_2)
    #print("==run X now==")
    #print(sess.run(X))

    #for step in range(steps):
    #    res2 = sess.run(assign_op2)
    #    print("%d step train" % (step + 1))
    #    print(res2)



my_cp_test(10)
#test_rand_list()
#cp_xxx_test()
#test_genABC([3,4,5],3)

#test_gen_test_tensor([2,3,4],2)
#test_assign_op()
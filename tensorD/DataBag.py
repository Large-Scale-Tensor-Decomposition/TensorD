#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/7/31 PM2:36
# @Author  : Shiloh Leung
# @Site    : 
# @File    : DataBag.py
# @Software: PyCharm Community Edition
import tensorflow as tf
import numpy as np
from tensorD.base.type import *

def primesfrom3to(n):
    """ Returns a array of primes, 3 <= p < n """
    sieve = np.ones(int(n/2), dtype=np.bool)
    for i in range(3,int(n**0.5)+1,2):
        if sieve[int(i/2)]:
            sieve[int(i*i/2)::i] = False
    return 2*np.nonzero(sieve)[0][1::]+1



def gen_core(coreR, number_list):
    """
    generate core tensor of shape R \\times R \\times R
    generate algrithm:
    return:
       tf.tensor of shape R \\times R \\times R
    """
    if coreR % 2 == 0:
        assert coreR*coreR/2 + coreR < len(number_list), "number list is not long enough"
    else:
        assert (coreR+1)*(coreR+1)/2 + coreR < len(number_list), "number list is not long enough"
    core = np.zeros(shape=(coreR, coreR, coreR))
    count = 0
    for dim1 in range(coreR):
        for dim2 in range(coreR):
            if dim1 % 2 == 0:
                row = number_list[count:count + coreR]
                if dim2 % 2 == 0:
                    core[:,dim2,dim1] = np.array(row)
                    count += 2
                else:
                    core[:,dim2,dim1] = np.array(row[::-1])
            else:
                core[:,:,dim1] = np.transpose(core[:,:,dim1-1])
    return core




def gen_ABC(I_list,R,number_list):
    """
    :param I_list: list of 3 int
    :param R: int, the shapes of A,B,C is I_i \\times R
    :param number_list: to choose from, list or np.ndarray
    :return:
      a list of 3 matrices, each shape is I_i \\times R
    """
    if isinstance(number_list, list):
        number_list = np.array(number_list)
    max_I = max(I_list)
    if max_I % 2 == 0:
        assert max_I/2 + R < len(number_list), "number list is not long enough"
    else:
        assert (max_I+1)/2 + R < len(number_list), "number list is not long enough"
    ABC = []
    for i in range(len(I_list)):
        count = 0
        ABC.append(np.zeros(shape=(I_list[i],R)))
        for row in range(I_list[i]):
            if row % 2 == 0:
                line = number_list[count:count + R]
                ABC[i][row,:] = line
                count += 1
            else:
                ABC[i][row,:] = line[::-1]
    return ABC


def gen_test_tensor(I_list, R, max_prime=10000):
    # the default number list has 1228 prime numbers, enough for R=48
    # but notice that max(I_list)/2 + R should be less than 1228
    number_list = primesfrom3to(max_prime)
    ABC = gen_ABC(I_list, R, number_list)
    core = gen_core(R, number_list)
    P = TTensor(core, ABC)
    with tf.Session() as sess:
        full_tensor = sess.run(P.extract())
    return full_tensor



def rand_list(shape_list, rank):
    # initial seed
    seed = 11796
    order = len(shape_list)
    init_mat = [np.zeros(shape=(shape_list[ii], rank)) for ii in range(order)]
    for mode in range(order):
        if mode != 0:
            I_i = shape_list[mode]
            for ii in range(I_i):
                for jj in range(rank):
                    seed = (8121*seed + 28411) % 1334456
                    init_mat[mode][ii,jj] = seed / 1334456 *10

    return init_mat



def insert_test(tf_variable):
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        print(sess.run(tf_variable))
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/7/14 PM8:53
# @Author  : Shiloh Leung
# @Site    : 
# @File    : DataBag.py
# @Software: PyCharm Community Edition
import tensorflow as tf
import numpy as np

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
                    core[dim1,dim2,:] = np.array(row)
                    count += 2
                else:
                    core[dim1,dim2,:] = np.array(row[::-1])
            else:
                core[dim1,:,:] = np.transpose(core[dim1-1,:,:])
    return core




def gen_ABC(I_list,R,number_list):
    """

    :param I_list: list of 3 int
    :param R: int
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








number_list = primesfrom3to(200)
print("core tensor:")
print(gen_core(3, number_list))
print("\n\nABC:")
print(gen_ABC([3,4,5], 3, number_list))




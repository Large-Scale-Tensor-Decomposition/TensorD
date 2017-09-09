import tensorflow as tf
import numpy as np

from tensorD.base.pitf_ops import *

init_op = tf.global_variables_initializer()


def generate_test():
    shape=tf.constant([3,4])
    rank = tf.constant(3)
    U,V = generate(shape,rank)
    return U,V


def centalizarion_test():
    mat = tf.random_normal((3,4))
    result = centralization(mat)
    return result


def subspace_test():
    shape = tf.constant([3, 4])
    rank = tf.constant(3)
    result = subspace(shape,rank,'A')
    #result = subspace(shape,rank,'B')
    #result = subspace(shape,rank,'C')
    return result


def sample_rule4mat_test():
    shape = tf.constant([3, 4, 5])
    ra = tf.constant(3)
    rb = tf.constant(4)
    rc = tf.constant(5)
    a,b,c = sample_rule4mat(shape,ra,rb,rc,10)
    return a,b,c

def sample3D_rule_test():
    shape = tf.constant([3, 4, 5])
    sample_num = 10
    a,b,c = sample3D_rule(shape,sample_num)
    return a,b,c


def Pomega_mat_test():
    shape = tf.constant([3, 4, 5])
    sp_num = 10
    a, b, c = sample3D_rule(shape, sp_num)
    spl = [a,b,c]
    mat1 = tf.random_normal((3,4))
    mat2 = tf.random_normal((4,5))
    mat3 = tf.random_normal((5,3))
    A = Pomega_mat(spl, mat1, shape, sp_num, dim=0)
    B = Pomega_mat(spl, mat2, shape, sp_num, dim=1)
    C = Pomega_mat(spl, mat3, shape, sp_num, dim=2)
    return A,B, C


def adjoint_operator_test():
    shape = tf.constant([3, 4, 5])
    sp_num = 10
    a, b, c = sample3D_rule(shape, sp_num)
    spl = [a, b, c]
    sp_vec = tf.random_uniform([sp_num])
    X = adjoint_operator(spl, sp_vec, shape, sp_num, dim=0)
    Y = adjoint_operator(spl, sp_vec, shape, sp_num, dim=1)
    Z = adjoint_operator(spl, sp_vec, shape, sp_num, dim=2)
    return X, Y, Z



def Pomega_tensor_test():
    shape = tf.constant([3, 4, 5])
    sp_num = 20
    a, b, c = sample3D_rule(shape, sp_num)
    spl = [a, b, c]
    tensor = tf.random_normal((3, 4, 5))
    sp_t = Pomega_tensor(spl,tensor, shape, sp_num)
    return sp_t



def Pomega_Pair_test():
    shape = tf.constant([3, 4, 5])
    sp_num = 10
    a, b, c = sample3D_rule(shape, sp_num)
    spl = [a, b, c]
    mat1 = tf.random_normal((3, 4))
    mat2 = tf.random_normal((4, 5))
    mat3 = tf.random_normal((5, 3))
    shape = tf.constant([3, 4, 5])
    sp_num = 10
    PA = Pomega_mat(spl,mat1,shape,sp_num,0)
    PB = Pomega_mat(spl, mat2, shape, sp_num, 1)
    PC = Pomega_mat(spl, mat3, shape, sp_num, 2)
    Pomega_Pair = PA+PB+PC
    return Pomega_Pair


def cone_projection_operator_test():#0.12 version doesn`t have norm function.
    xx=tf.random_normal([5])
    tt=tf.constant(1)
    t1,t2=cone_projection_operator(xx,tt)
    return t1,t2


def SVT_test():
    shape = tf.constant([3, 4, 5])
    mat1 = tf.random_normal((3, 4))
    tao = tf.constant(0.0)
    s, u, v = SVT(mat1, tao)
    return s, u, v


def shrink_test():#false
    shape = tf.constant([3, 4, 5])
    mat1 = tf.random_normal((3, 4))
    tao = tf.constant(0.0)
    print('matrix shape:', mat1.get_shape().as_list())
    tmp_normal = shrink(mat1,tao,mode='normal')
    tmp_complicated = shrink(mat1,tao,mode='complicated')
    return tmp_normal,tmp_complicated


with tf.Session() as sess:
    sess.run(init_op)
    # a,b,c = Pomega_mat_test()
    # a, b, c = adjoint_operator_test()
    # ss = Pomega_tensor_test()
    # p = Pomega_Pair_test()
    n, c = shrink_test()
    # s,u,v = SVT_test()
    # print(sess.run(a))
    # print(sess.run(b))
    # print(sess.run(c))
    # print(sess.run(ss))
    print(sess.run(n))
    print(sess.run(c))
    sess.close()


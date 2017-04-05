# Created by ay27 at 17/3/14
import pickle
import numpy as np
from tensorD.base.type import KTensor
import tensorflow as tf
import tensorD.loss as validator

tmp = []
for i in range(3):
    file = open('w_%d' % i, 'rb')
    tmp.append(pickle.load(file))

# for ii in range(len(tmp[0])):
#     print(tmp[0][ii], tmp[1][ii], tmp[2][ii])

A = []
B = []
C = []
for i in range(3):
    file = open('wa_%d' % i, 'rb')
    A.append(pickle.load(file))

    file = open('wb_%d' % i, 'rb')
    B.append(pickle.load(file))

    file = open('wc_%d' % i, 'rb')
    C.append(pickle.load(file))

I = 3
J = 4
K = 5
STEP = 1000
# tensor = np.random.rand(I, J, K)
tensor = tf.constant(np.arange(I * J * K).reshape(I, J, K), dtype=tf.float64)

with tf.Session() as sess:
    for i in range(3):
        t = KTensor([A[i], B[i], C[i]])

        print(sess.run(validator.rmse(tensor - t.extract())))

        print(sess.run(tensor - t.extract()))
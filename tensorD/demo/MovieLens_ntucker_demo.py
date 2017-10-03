#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/3 PM4:16
# @Author  : Shiloh Leung
# @Site    : 
# @File    : MovieLens_ntucker_demo.py
# @Software: PyCharm Community Edition

from tensorD.demo.MovieLens_process import *
from tensorD.dataproc.reader import TensorReader
import tensorflow as tf

if __name__ == '__main__':
    csv_file = 'movielens_data.csv'
    #movie_dict = data_write(csv_file)
    reader = TensorReader(csv_file)
    start = time.time()
    reader.read()
    end = time.time()
    print('Read time: %.6f s\n' % (end - start))
    with tf.Session() as sess:
        rating_tensor = sess.run(reader.full_data)



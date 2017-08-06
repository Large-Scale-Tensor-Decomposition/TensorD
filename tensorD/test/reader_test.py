#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/8/6 PM3:31
# @Author  : Shiloh Leung
# @Site    : 
# @File    : reader_test.py
# @Software: PyCharm Community Edition

from tensorD.dataproc.reader import TensorReader

if __name__ == '__main__':
    print('csv file:')
    file_path1 = 'data1.csv'
    tensor_reader1 = TensorReader(file_path1)
    tensor_reader1.read()

    print('\ntxt file:')
    file_path2 = 'data1.txt'
    tensor_reader2 = TensorReader(file_path2)
    tensor_reader2.read()


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
    file_path = 'data1.csv'
    treader = TensorReader(file_path)
    treader.read()
    print(treader.full_data)
    print(treader.sparse_data)


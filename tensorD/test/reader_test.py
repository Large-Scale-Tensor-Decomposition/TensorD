#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/8/6 PM3:31
# @Author  : Shiloh Leung
# @Site    : 
# @File    : reader_test.py
# @Software: PyCharm Community Edition

from tensorD.dataproc.reader import TensorReader
import time

if __name__ == '__main__':
    print('csv file:')
    file_path = 'data1.csv'

    treader1 = TensorReader(file_path)
    start1 = time.time()
    treader1.read_demo1()
    end1 = time.time()
    print('read demo 1 time: %.6f s\n' % (end1 - start1))

    treader2 = TensorReader(file_path)
    start2 = time.time()
    treader2.read_demo2()
    end2 = time.time()
    print('read demo 2 time: %.6f s\n' % (end2 - start2))

    treader3 = TensorReader(file_path)
    start3 = time.time()
    treader3.read_demo3()
    end3 = time.time()
    print('read demo 3 time: %.6f s\n' % (end3 - start3))

    treader4 = TensorReader(file_path)
    start4 = time.time()
    treader4.read_demo4()
    end4 = time.time()
    print('read demo 4 time: %.6f s\n' % (end4 - start4))

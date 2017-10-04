#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/4 PM2:30
# @Author  : Shiloh Leung
# @Site    : 
# @File    : MovieLens_data_make.py
# @Software: PyCharm Community Edition
from tensorD.demo.MovieLens_process import *

if __name__ == '__main__':
    u_data_make()
    u_csv_make('ml-100k/u1.base', 'u1.base.csv')
    u_csv_make('ml-100k/u2.base', 'u2.base.csv')
    u_csv_make('ml-100k/u3.base', 'u3.base.csv')
    u_csv_make('ml-100k/u4.base', 'u4.base.csv')
    u_csv_make('ml-100k/u5.base', 'u5.base.csv')
    u_csv_make('ml-100k/u1.test', 'u1.test.csv')
    u_csv_make('ml-100k/u2.test', 'u2.test.csv')
    u_csv_make('ml-100k/u3.test', 'u3.test.csv')
    u_csv_make('ml-100k/u4.test', 'u4.test.csv')
    u_csv_make('ml-100k/u5.test', 'u5.test.csv')
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/2 PM4:59
# @Author  : Shiloh Leung
# @Site    : 
# @File    : MovieLens_process.py
# @Software: PyCharm Community Edition

"""
MovieLens data set

    Max user ID: 671
    Max movie ID: 163949 , miss 154883

"""

import csv
import numpy as np
import time



def time_month_group(now_timestamp, start_timestamp):
    """
        Transform timestamp into ``month_group``. The ``month_group`` of ``start_timestamp`` is ``0``.

        Parameters
        ----------
        now_timestamp : int
            timestamp to transform
        start_timestamp : int
            timestamp of the beginning

        Returns
        -------
        int

    """
    start_time = time.localtime(start_timestamp)
    now_time = time.localtime(now_timestamp)
    month_group = 12 * (now_time.tm_year - start_time.tm_year) + now_time.tm_mon - start_time.tm_mon
    return month_group


def u_data_make():
    file_path = 'ml-100k/u.data'
    file = open(file_path, 'r')
    all_data = []
    all_rating = []
    start_timestamp = 1000000000000
    for str_row in csv.reader(file):
        row = str_row[0].split()    # each row is 'user id | item id | rating | timestamp'
        all_data.append([int(row[0]), int(row[1]), int(row[3])])
        all_rating.append(float(row[2]))
        if int(row[3]) < start_timestamp:
            start_timestamp = int(row[3])
    file.close()
    for ii in range(100000):
        all_data[ii][2] = time_month_group(all_data[ii][2], start_timestamp)    # start_timestamp = 874724710
    with open('u_data.csv', 'w') as out_file:
        writer = csv.writer(out_file)
        for ii in range(100000):
            user_id = str(all_data[ii][0]-1)    # user_id = 0 ~ 942
            movie_id = str(all_data[ii][1]-1)    # movie_id = 0 ~ 1681
            month_group = str(all_data[ii][2])    # month_group = 0 ~ 7
            rating = str(all_rating[ii])
            writer.writerow([user_id, movie_id, month_group, rating])


def u_csv_make(u_base_path, out_path):
    """
    Transform original *.base file or *.test file into csv file for TensorReader,
    each row of new csv file is ``user_id, movie_id, month_group, rating``.
    Considering the rating data starts from September 1997 and ends in April 1998,
    the ``month_group`` ranges from ``0`` to ``7``.

        Parameters
        ----------
        u_base_path : str
            name of *.base file or *.test file
        out_path : str
            name of csv file

    """
    u_base_file = open(u_base_path, 'r')
    all_data = []
    all_rating = []
    start_timestamp = 874724710
    for str_row in csv.reader(u_base_file):
        row = str_row[0].split()    # each row is 'user id | item id | rating | timestamp'
        all_data.append([int(row[0]), int(row[1]), time_month_group(int(row[3]), start_timestamp)])
        all_rating.append(float(row[2]))
        if int(row[3]) < start_timestamp:
            start_timestamp = int(row[3])
    u_base_file.close()
    iter_count = len(all_rating)
    with open(out_path, 'w') as out_file:
        writer = csv.writer(out_file)
        for ii in range(iter_count):
            user_id = str(all_data[ii][0]-1)    # user_id = 0 ~ 942
            movie_id = str(all_data[ii][1]-1)    # movie_id = 0 ~ 1681
            month_group = str(all_data[ii][2])    # month_group = 0 ~ 7
            rating = str(all_rating[ii])
            writer.writerow([user_id, movie_id, month_group, rating])











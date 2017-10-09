#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/2 PM4:59
# @Author  : Shiloh Leung
# @Site    : 
# @File    : MovieLens_process.py
# @Software: PyCharm Community Edition

"""
MovieLens data set

    Max user ID:
    Max movie ID:

"""

import csv
import numpy as np
import time
import datetime



def time_week_group(now_timestamp, start_timestamp):
    """
        Transform timestamp into ``week_group``. The ``week_group`` of ``start_timestamp`` is ``0``.

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
    start_day = datetime.datetime(start_time.tm_year, start_time.tm_mon, start_time.tm_mday)
    now_day = datetime.datetime(now_time.tm_year, now_time.tm_mon, now_time.tm_mday)
    week_group = np.floor((now_day - start_day).days / 7)
    return int(week_group)

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
    month_group = 12*(now_time.tm_year-start_time.tm_year) + now_time.tm_mon-start_time.tm_mon
    return int(month_group)


def ml_1m_data():
    file_path = 'ratings.csv'
    start_timestamp = 789652009
    with open(file_path, 'r') as in_file:
        reader = csv.reader(in_file)
        str_in = [row for row in reader]
    str_in = str_in[1::]
    tmp_set = set([int(row[1]) for row in str_in])
    movie_count = 9066    # 9066 movies in all
    movie_id_set = sorted(list(tmp_set))
    movie_dict = dict()
    for ii in range(movie_count):
        movie_id = movie_id_set[ii]
        movie_dict[movie_id] = ii    # {movie_id:movie_idx}
    with open('new_ratings_all.csv', 'w') as out_file:
        writer = csv.writer(out_file)
        rating_count = len(str_in)
        all_data = []
        for ii in range(rating_count):
            user_id = int(str_in[ii][0]) - 1    # user_id = 0~670
            movie_id = int(str_in[ii][1])
            movie_idx = movie_dict[movie_id]    # movie_idx = 0~9065
            month_group = time_month_group(int(str_in[ii][3]), start_timestamp)    # month_group = 0~261
            rating = float(str_in[ii][2])
            all_data.append([user_id, movie_idx, month_group, rating])
            writer.writerow([str(user_id), str(movie_idx), str(month_group), str(rating)])
    return all_data




def data_random():
    # [943, 1682, 31]
    with open('random_data.csv', 'w') as out_file:
        writer = csv.writer(out_file)
        for user_id in range(600):
            for movie_id in range(1500):
                for month_group in range(30):
                    rand_seed = np.random.rand()
                    if rand_seed < 0.01:
                        rating = np.ceil(5*np.random.rand())
                        writer.writerow([str(user_id), str(movie_id), str(month_group), str(rating)])






def u_data_make():
    file_path = 'ml-100k/u.data'
    start_timestamp = 874724710

    with open(file_path, 'r') as in_file:
        reader = csv.reader(in_file, delimiter='\t')
        all_data = [[int(row[0])-1, int(row[1])-1, time_week_group(int(row[3]), start_timestamp), float(row[2])] for row in reader]

    with open('movielens-100k/u.data.csv', 'w') as out_file:
        writer = csv.writer(out_file)
        for ii in range(100000):
            user_id = str(all_data[ii][0])    # user_id = 0 ~ 942
            movie_id = str(all_data[ii][1])    # movie_id = 0 ~ 1681
            week_group = str(all_data[ii][2])    # month_group = 0 ~ 30
            rating = str(all_data[ii][3])
            writer.writerow([user_id, movie_id, week_group, rating])


def u_csv_make(in_path, out_path):
    """
    Transform original *.base file or *.test file into csv file for TensorReader,
    each row of new csv file is ``user_id, movie_id, week_group, rating``.
    Considering the rating data starts from September 19th, 1997 through April 22nd, 1998,
    the ``week_group`` ranges from ``0`` to ``31``.

        Parameters
        ----------
        in_path : str
            name of *.base file or *.test file
        out_path : str
            name of csv file

    """
    start_timestamp = 874724710

    with open(in_path, 'r') as in_file:
        reader = csv.reader(in_file, delimiter='\t')
        all_data = [[int(row[0]) - 1, int(row[1]) - 1, time_week_group(int(row[3]), start_timestamp), float(row[2])] for
                    row in reader]
    rating_count = len(all_data)
    with open(out_path, 'w') as out_file:
        writer = csv.writer(out_file)
        for ii in range(rating_count):
            user_id = str(all_data[ii][0])  # user_id = 0 ~ 942
            movie_id = str(all_data[ii][1])  # movie_id = 0 ~ 1681
            week_group = str(all_data[ii][2])  # month_group = 0 ~ 30
            rating = str(all_data[ii][3])
            writer.writerow([user_id, movie_id, week_group, rating])

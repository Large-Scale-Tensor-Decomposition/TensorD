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

def data_write(out_file_path):
    """
    Transform original ratings.csv into csv file for TensorReader,
    each row of new csv file is ``user_id, movie_idx, month_group, rating``.
    Return movie index to which the movie ID corresponds.
    Considering the rating data starts from January 1995 and ends in October 2016,
    the ``month_group`` ranges from ``0`` to ``261``.

        Parameters
        ----------
        out_file_path : str
            name of csv file

        Returns
        -------
        movie_dict : dict
            a dictionary of movie ID and movie index, each element is ``movie_id:movie_idx``
    """
    file_path = 'MovieLens/ml-latest-small/ratings.csv'
    file = open(file_path, 'r')
    str_in = []
    for row in csv.reader(file):
        str_in.append(row)
    file.close()

    movie_set = sorted(list(set([int(rating[1]) for rating in str_in[1::]])))
    movie_dict = dict()
    for ii in range(len(movie_set)):
        movie_id = movie_set[ii]
        movie_dict[movie_id] = ii  # format in movie_dict: {movie_id:movie_idx}

    with open(out_file_path, 'w') as out_file:
        writer = csv.writer(out_file)
        for rating in str_in[1::]:
            # userId,movieId,rating,timestamp
            user_id = str(int(rating[0]) - 1)
            movie_idx = str(movie_dict[int(rating[1])])
            score = rating[2]
            month_group = str(time_month_group(int(rating[3]), 789652009))    # timestamp start from 789652009
            writer.writerow([user_id, movie_idx, month_group, score])
    return movie_dict


def time_month_group(now_timestamp, start_timestamp):
    start_time = time.localtime(start_timestamp)
    now_time = time.localtime(now_timestamp)
    month_group = 12 * (now_time.tm_year - start_time.tm_year) + now_time.tm_mon - start_time.tm_mon
    return month_group

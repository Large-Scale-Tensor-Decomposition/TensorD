# # Created by ay27 at 17/2/7
# import parse
#
import csv
import numpy as np
import tensorflow as tf
from ctypes import ArgumentError


class TensorReader(object):
    """

    Attibutes
    ---------

    """
    def __init__(self, file_path):
        """

        Parameters
        ----------
        file_path: file path
        type: type of file, default 'csv'
        """
        self._file_path = file_path
        self._dense = None
        self._sparse = None
        self._type = self._file_path.split('.')[-1]


    def read(self):
        file = open(self._file_path, 'r')
        str_in = []
        if self._type == 'csv' or self._type == 'txt':
            for row in csv.reader(file):
                if len(row) != 0:
                    str_in.append(row)
        else:
            raise ArgumentError(self._type + ' file is not supported by TensorReader.')
        print(str_in)


        order = len(str_in[0]) - 1
        sparse_data = dict()
        max_dim = [0 for _ in range(order)]

        for entry in str_in:
            idx_tuple = tuple([int(entry[mode]) for mode in range(order)])
            sparse_data[idx_tuple] = float(entry[-1])

            # TODO : assume that index starts from zero ? can be selected later
            for mode in range(order):
                if idx_tuple[mode] > max_dim[mode]:
                    max_dim[mode] = idx_tuple[mode]


        self._sparse = sparse_data
        print(sparse_data)
        print(max_dim)
















#
# class TensorReader(object):
#     """
#
#     Attributes
#     ----------
#     file: opening file with given file_path
#
#     """
#
#     def __init__(self, file_path, fmt='{:d} {:d} {:d} {:f}', encoding=None):
#         """
#
#         Parameters
#         ----------
#         file_path: file path
#         fmt: line format of data, see parse package_ for more details
#         encoding: file encoding
#
#         .. _parse package: https://pypi.python.org/pypi/parse
#         """
#         self.file = open(file_path, mode='r', encoding=encoding)
#         self.fmt = parse.compile(fmt)
#
#     def next(self):
#         """
#         Get the next line data.
#
#         Yields
#         ------
#         list
#             list of one row data
#         """
#         for line in self.file:
#             if not line:
#                 continue
#             items = list(self.fmt.parse(line))
#             if items:
#                 yield items

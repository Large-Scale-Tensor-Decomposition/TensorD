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
        self._sparse_data = None
        self._full_data = None
        self._type = self._file_path.split('.')[-1]

    def read_demo1(self):
        file = open(self._file_path, 'r')
        str_in = []
        if self._type == 'csv' or self._type == 'txt':
            for row in csv.reader(file):
                if len(row) != 0:
                    str_in.append(row)
        else:
            raise ArgumentError(self._type + ' file is not supported by TensorReader.')
        file.close()

        order = len(str_in[0]) - 1
        entry_count = len(str_in)
        value = np.zeros(entry_count)
        idx = np.zeros(shape=(entry_count, order), dtype=int)
        for row in range(entry_count):
            entry = str_in[row]
            idx[row] = np.array([int(entry[mode]) for mode in range(order)])
            value[row] = float(entry[-1])
        max_dim = np.max(idx, axis=0) + np.ones(order).astype(int)
        self._sparse_data = tf.SparseTensor(indices=idx, values=value, dense_shape=max_dim)
        self._full_data = tf.sparse_tensor_to_dense(self._sparse_data, validate_indices=False)

    def read_demo2(self):
        tmp = np.loadtxt(self._file_path, dtype=float, delimiter=',')
        order = len(tmp[0]) - 1
        max_dim = np.max(tmp[:, 0:order], axis=0).astype(int) + np.ones(order).astype(int)
        self._sparse_data = tf.SparseTensor(indices=tmp[:, 0:order].astype(int), values=tmp[:, order],
                                            dense_shape=max_dim)
        self._full_data = tf.sparse_tensor_to_dense(self._sparse_data, validate_indices=False)

    def read_demo3(self):
        file = open(self._file_path, 'r')
        str_in = []
        if self._type == 'csv' or self._type == 'txt':
            for row in csv.reader(file):
                if len(row) != 0:
                    str_in.append(row)
        else:
            raise ArgumentError(self._type + ' file is not supported by TensorReader.')
        file.close()

        idx = []
        value = []
        order = len(str_in[0]) - 1
        max_dim = [0 for _ in range(order)]

        for entry in str_in:
            tmp_idx = [int(entry[mode]) for mode in range(order)]
            idx.append(tmp_idx)
            value.append(float(entry[-1]))
            # TODO : assume that index starts from zero ? can be selected later
            for mode in range(order):
                if max_dim[mode] < tmp_idx[mode] + 1:
                    max_dim[mode] = tmp_idx[mode] + 1

        self._sparse_data = tf.SparseTensor(indices=idx, values=value, dense_shape=max_dim)
        self._full_data = tf.sparse_tensor_to_dense(self._sparse_data, validate_indices=False)

    def read_demo4(self):
        file = open(self._file_path, 'r')
        str_in = []
        if self._type == 'csv' or self._type == 'txt':
            for row in csv.reader(file):
                if len(row) != 0:
                    str_in.append(row)
        else:
            raise ArgumentError(self._type + ' file is not supported by TensorReader.')

        file.close()

        order = len(str_in[0]) - 1
        sparse_data = dict()
        max_dim = [0 for _ in range(order)]

        for entry in str_in:
            idx_tuple = tuple([int(entry[mode]) for mode in range(order)])
            sparse_data[idx_tuple] = float(entry[-1])

            # TODO : assume that index starts from zero ? can be selected later
            for mode in range(order):
                if max_dim[mode] < idx_tuple[mode] + 1:
                    max_dim[mode] = idx_tuple[mode] + 1

        full_data = np.zeros(shape=max_dim)
        for entry in sparse_data:
            full_data[entry] = sparse_data[entry]

        self._sparse_data = sparse_data
        self._full_data = full_data

    @property
    def full_data(self):
        return self._full_data

    @property
    def sparse_data(self):
        return self._sparse_data

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

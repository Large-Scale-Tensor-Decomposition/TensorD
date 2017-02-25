# Created by ay27 at 17/2/7
import numpy as np


class Provider(object):
    def next_batch(self, size):
        pass


class OrdProvider(Provider):
    """
    Data Provider, split data in given order(mode).
    """

    def __init__(self, reader, order, task_cnt=1, task_index=0, batch_size=1, sparse=False, shape=None):
        self.reader = reader
        self.order = order
        self.task_index = task_index
        self.task_cnt = task_cnt
        self.is_sparse = sparse
        self.shape = shape
        self.batch_size = batch_size

        # store in dense
        self.data = None

        if self.is_sparse:
            self._read_sparse()
        else:
            self._read_dense()

        self._split_size = int(self.shape[self.order] / self.task_cnt)

        self._offset = self.task_index * self._split_size

    def __iter__(self):
        return self

    def __next__(self):
        """

        Yields
        ------
        ndarray
            batch of data
        """
        cur_index = 0
        while cur_index < self._split_size:
            end = min(cur_index + self.batch_size, self._split_size)
            yield self.data[cur_index:end]
            cur_index += self.batch_size
        raise StopIteration()

    def _read_sparse(self):
        input_data = np.asarray([row for row in self.reader.next()])
        if not self.shape:
            self.shape = [np.max(input_data, axis) for axis in range(len(input_data[0]))]

        split_shape = self.shape.copy()
        split_shape[self.order] = self._split_size
        self.data = np.zeros(split_shape)
        for row in input_data:
            if self._offset <= row[self.order] < self._offset + self._split_size:
                row[self.order] -= self._offset
                self.data.itemset(row[:-1], row[-1])

    def _read_dense(self):
        self.data = np.asarray(
            [row for (i, row) in enumerate(self.reader.next()) if
             self._offset <= i < self._offset + self._split_size])
        if not self.shape:
            self.shape = self.data.shape

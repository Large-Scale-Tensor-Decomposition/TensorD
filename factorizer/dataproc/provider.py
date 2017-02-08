# Created by ay27 at 17/2/7
import numpy as np


class Provider(object):
    def next_batch(self, size):
        pass


class OrdProvider(Provider):
    """
    Data Provider, split data in given order(mode).
    """

    def __init__(self, reader, order, task_id=0, split_cnt=1, sparse=False, shape=None):
        self._reader = reader
        self._order = order
        self._task_id = task_id
        self._split_cnt = split_cnt
        self._sparse = sparse
        self.shape = shape

        # store in dense
        self.data = None

        if self._sparse:
            self._read_sparse()
        else:
            self._read_dense()

        self._split_size = int(self.shape[self._order] / self._split_cnt)

        self._offset = self._task_id * self._split_size

    def next_batch(self, size):
        """

        Parameters
        ----------
        size: batch size

        Yields
        ------
        ndarray
            batch of data
        """
        cur_index = 0
        while cur_index < self._split_size:
            end = min(cur_index + size, self._split_size)
            yield self.data[cur_index:end]
            cur_index += size
        raise StopIteration()

    def _read_sparse(self):
        input_data = np.asarray([row for row in self._reader.next()])
        if not self.shape:
            self.shape = [np.max(input_data, axis) for axis in range(len(input_data[0]))]

        split_shape = self.shape.copy()
        split_shape[self._order] = self._split_size
        self.data = np.zeros(split_shape)
        for row in input_data:
            if self._offset <= row[self._order] < self._offset + self._split_size:
                row[self._order] -= self._offset
                self.data.itemset(row[:-1], row[-1])

    def _read_dense(self):
        self.data = np.asarray(
            [row for (i, row) in enumerate(self._reader.next()) if
             self._offset <= i < self._offset + self._split_size])
        if not self.shape:
            self.shape = self.data.shape

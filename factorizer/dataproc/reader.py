# Created by ay27 at 17/2/7
import parse


class TensorReader(object):
    """

    Attributes
    ----------
    file: opening file with given file_path

    """
    def __init__(self, file_path, fmt='{:f} {:f} {:f} {:f}', encoding=None):
        """

        Parameters
        ----------
        file_path: file path
        fmt: line format of data, see parse package_ for more details
        encoding: file encoding

        .. _parse package: https://pypi.python.org/pypi/parse
        """
        self.file = open(file_path, mode='r', encoding=encoding)
        self.fmt = parse.compile(fmt)

    def next(self):
        """
        Get the next line data.

        Yields
        ------
        list
            list of one row data
        """
        for line in self.file:
            items = list(self.fmt.parse(line))
            if items:
                yield items
        yield StopIteration
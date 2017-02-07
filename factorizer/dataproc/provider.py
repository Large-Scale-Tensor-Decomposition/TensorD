# Created by ay27 at 17/2/7


class Provider(object):
    def next_batch(self, size):
        pass


class OrdProvider(Provider):
    """
    Data Provider, split data in given order(mode).
    """
    def __init__(self, reader, order):
        pass
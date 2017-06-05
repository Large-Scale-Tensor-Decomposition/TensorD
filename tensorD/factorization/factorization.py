# Created by ay27 at 17/6/2


class Model(object):
    def __init__(self, train_op, loss_op, var_list, init_op, input_data, args):
        self.train_op = train_op
        self.loss_op = loss_op
        self.var_list = var_list
        self.init_op = init_op
        self.input_data = input_data
        self.args = args


class BaseFact(object):
    def build_model(self):
        raise NotImplementedError

    def predict(self, key):
        raise NotImplementedError

    def train(self, steps):
        raise NotImplementedError

    def full(self):
        raise NotImplementedError

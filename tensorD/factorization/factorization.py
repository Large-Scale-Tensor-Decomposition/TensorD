# Created by ay27 at 17/6/2
import pickle


class Model(object):
    def __init__(self, env, train_op, loss_op, var_list, init_op, full_tensor_op, args):
        self.env = env
        self.train_op = train_op
        self.loss_op = loss_op
        self.var_list = var_list
        self.init_op = init_op
        self.full_tensor_op = full_tensor_op
        self.args = args

    def save(self, save_path):
        pickle.dump(self, open(save_path, 'wb'))

    @staticmethod
    def load(save_path):
        return pickle.load(open(save_path, 'rb'))


class BaseFact(object):
    def build_model(self, args) -> Model:
        raise NotImplementedError

    def predict(self, key):
        raise NotImplementedError

    def train(self, steps):
        raise NotImplementedError

    def full(self):
        raise NotImplementedError

# Created by ay27 at 17/6/2
import pickle


class Model(object):
    """
    The Model class holding the
    """

    def __init__(self, before_train, in_train, after_train=None, metrics=None):
        self.before_train = before_train    # containing env, init_op, norm_input_op, args
        self.in_train = in_train    # containing train_op
        self.metrics = metrics    # containing fit_op_not_zero, fit_op_zero, loss_op
        self.after_train = after_train    # containing full_op_final, var_list_final

        # TODO : how to save and restore the model properly
        # def save(self, save_path):
        #     pickle.dump(self, open(save_path, 'wb'))
        #
        # @staticmethod
        # def load(save_path):
        #     return pickle.load(open(save_path, 'rb'))


class BaseFact(object):
    def build_model(self, args) -> Model:
        raise NotImplementedError

    def train(self, steps=None):
        raise NotImplementedError

    def predict(self, *key):
        raise NotImplementedError

    def full(self):
        raise NotImplementedError

    def save(self, path):
        # TODO
        pass

    @staticmethod
    def restore(path):
        # TODO
        pass

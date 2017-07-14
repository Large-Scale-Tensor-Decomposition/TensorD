# Created by ay27 at 17/1/13
import numpy as np
import tensorflow as tf
from tensorD.base import *
from tensorD.loss import *
from numpy.random import rand
from .factorization import Model, BaseFact
from .env import Environment


class CP_ALS(BaseFact):
    class CP_Args(object):
        def __init__(self,
                     rank,
                     tol=10e-6,
                     validation_internal=-1,
                     get_lambda=False,
                     get_rmse=False,
                     verbose=False):
            self.rank = rank
            self.tol = tol
            self.validation_internal = validation_internal
            self.get_lambda = get_lambda
            self.get_rmse = get_rmse
            self.verbose = verbose

    def __init__(self, env):
        assert isinstance(env, Environment)
        self._env = env
        self._model = None
        self._full_tensor = None
        self._is_train_finish = False

    def build_model(self, args) -> Model:
        assert isinstance(args, CP_ALS.CP_Args)
        return self._build_single_model(args)

    def predict(self, *key):
        if not self._full_tensor:
            raise TensorErr('improper stage to call predict before the model is trained')
        return self._full_tensor.item(key)

    def train(self, steps=None):
        self._train_in_single(steps)

    def full(self):
        return self._full_tensor

    def train_finish(self):
        return self._is_train_finish

    def factors(self):
        return self._factors


    def _build_single_model(self, args):
        input_data = self._env.full_data()
        shape = input_data.get_shape().as_list()
        order = len(shape)
        #A = [tf.Variable(rand(shape[ii], args.rank), name='A-%d' % ii) for ii in range(order)]
        # TODO : Fix the initial random matrices
        A = []
        U1 = np.array(np.zeros(shape=(10,10)))

        A.append(tf.Variable(U1, name='A-1'))
        A.append(tf.Variable(U2, name='A-2'))


        mats = [ops.unfold(input_data, mode) for mode in range(order)]

        assign_op = [None for _ in range(order)]
        for mode in range(order):
            AtA = [tf.matmul(A[ii], A[ii], transpose_a=True, name='AtA-%d-%d' % (mode,ii)) for ii in range(order)]
            V = ops.hadamard(AtA, skip_matrices_index=mode)
            XA = tf.matmul(mats[mode], ops.khatri(A, mode, True), name='XA-%d' % mode)

            # TODO : does it correct to assign A[mode] with the **assign op** ?
            # But it's strangle to raise not invertible when remove the A[mode] assignment.
            assign_op[mode] = A[mode] = A[mode].assign(
                tf.transpose(tf.matrix_solve(tf.transpose(V), tf.transpose(XA)), name='TTT-%d' % mode))

        P = KTensor(A)
        full_op = P.extract()
        # TODO : does it correct?
        loss_op = rmse_ignore_zero(input_data, full_op)

        # TODO : should I use constant in that way?
        """
        fitness = 1 - \frac{\left \| X - X_{real}  \right \|_F}{\left \| X  \right \|_F}
        """
        one = tf.constant(1,dtype=loss_op.dtype)
        fit_op =  one - l2(input_data, full_op)/tf.norm(input_data)


        tf.summary.scalar('loss', loss_op)
        tf.summary.scalar('fitness', fit_op)

        train_op = tf.group(*assign_op)
        var_list = A

        init_op = tf.global_variables_initializer()

        self._model = Model(self._env, train_op, loss_op, fit_op, var_list, init_op, full_op, args)
        return self._model

    def _train_in_single(self, steps):
        self._is_train_finish = False
        self._full_tensor = None

        sess = self._env.sess
        model = self._model
        args = model.args

        sum_op = tf.summary.merge_all()
        sum_writer = tf.summary.FileWriter(self._env.summary_path, sess.graph)

        sess.run(model.init_op)

        print('CP model initial finish')
        for step in range(steps):
            sess.run(model.train_op)
            if step + 1 == steps:
                loss_v, self._full_tensor = sess.run([model.loss_op, model.full_tensor_op])
                sum_writer.add_summary(sess.run(sum_op), step)
                self._factors = sess.run(model.var_list)
                print('step=%d, RMSE=%f, fit=%f' % (step, loss_v, fitness))

            elif args.verbose or step == 0 or step % args.validation_internal == 0:
                loss_v = sess.run(model.loss_op)
                fitness = sess.run(model.fit_op)
                sum_writer.add_summary(sess.run(sum_op), step)
                print('step=%d, RMSE=%f, fit=%f' % (step, loss_v, fitness))

        print('CP model train finish, with RMSE = %f, fit=%f' % (loss_v, fitness))
        self._is_train_finish = True

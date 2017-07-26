# Created by ay27 at 17/1/13
import numpy as np
import tensorflow as tf
from tensorD.base import *
from tensorD.loss import *
from numpy.random import rand
from .factorization import Model, BaseFact
from .env import Environment
from DataBag import *


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
        self._lambdas = None



    def predict(self, *key):
        if not self._full_tensor:
            raise TensorErr('improper stage to call predict before the model is trained')
        return self._full_tensor.item(key)

    def train(self, steps=None):
        self._train_in_single(steps)

    @property
    def full(self):
        return self._full_tensor

    @property
    def train_finish(self):
        return self._is_train_finish

    @property
    def factors(self):
        return self._factors

    @property
    def lambdas(self):
        return self._lambdas


    def build_model(self, args):
        assert isinstance(args, CP_ALS.CP_Args)
        input_data = self._env.full_data()
        shape = input_data.get_shape().as_list()
        order = len(shape)
        # TODO : Fix the initial random matrices ?
        with tf.name_scope('random-initial') as scope:
            #A = [tf.Variable(rand(shape[ii], args.rank), name='A-%d' % ii) for ii in range(order)]
            Uinit = rand_list(shape, args.rank)
            A = [tf.Variable(Uinit[ii], name='A_1_iter-%d' % ii) for ii in range(order)]
            mats = [ops.unfold(input_data, mode) for mode in range(order)]
            assign_op = [None for _ in range(order)]


        for mode in range(order):
            if mode != 0:
                with tf.control_dependencies([assign_op[mode - 1]]):
                    AtA = [tf.matmul(A[ii], A[ii], transpose_a=True, name='AtA-%d-%d' % (mode,ii)) for ii in range(order)]
                    XA = tf.matmul(mats[mode], ops.khatri(A, mode, True), name='XA-%d' % mode)
            else:
                AtA = [tf.matmul(A[ii], A[ii], transpose_a=True, name='AtA-%d-%d' % (mode, ii)) for ii in range(order)]
                XA = tf.matmul(mats[mode], ops.khatri(A, mode, True), name='XA-%d' % mode)

            V = ops.hadamard(AtA, skip_matrices_index=mode)
            non_norm_A = tf.matmul(XA, tf.py_func(np.linalg.pinv, [V], tf.float64),name='XAV-%d' % mode)
            lambda_op = tf.reduce_max(tf.reshape(non_norm_A,shape=(shape[mode], args.rank)), axis=0)
            assign_op[mode] = A[mode].assign(tf.div(non_norm_A, lambda_op))

        train_op = tf.group(*assign_op)

        with tf.name_scope('full-tensor-in-train') as scope:
            #with tf.control_dependencies([assign_op[order-1]]):
            P = KTensor(assign_op, lambda_op)
            full_op = P.extract()
            insert_test(full_op)



        with tf.name_scope('loss-in-train') as scope:
            loss_op = rmse_ignore_zero(input_data, full_op)

        """
        fitness = 1 - \\frac{\\left \\| X - X_{real}  \\right \\|_F}{\\left \\| X  \\right \\|_F}
        """
        with tf.name_scope('fitness-in-train') as scope:
            norm_input_data = tf.norm(input_data)
            fit_op_not_zero = 1 - tf.sqrt(tf.square(norm_input_data) + tf.square(tf.norm(full_op)) - 2*ops.inner(input_data, full_op)) / norm_input_data
            #fit_op_zero = tf.square(tf.norm(full_op)) - 2 * ops.inner(input_data, full_op)


        tf.summary.scalar('loss', loss_op)
        tf.summary.scalar('fitness', fit_op_not_zero)
        #tf.summary.scalar('fitness_0', fit_op_zero)

        init_op = tf.global_variables_initializer()

        before_train = [self._env, args, init_op, norm_input_data]
        in_train = [assign_op, lambda_op, full_op, train_op, A]
        after_train = [None]
        metrics = [None, fit_op_not_zero, loss_op]

        self._model = Model(before_train, in_train, after_train, metrics)
        return self._model

    def _train_in_single(self, steps):
        self._is_train_finish = False
        self._full_tensor = None

        sess = self._env.sess
        model = self._model
        args = model.before_train[1]
        init_op = model.before_train[2]

        assign_op = model.in_train[0]
        lambda_op = model.in_train[1]
        full_op = model.in_train[2]
        train_op = model.in_train[3]
        A = model.in_train[4]

        #fit_op_zero = model.metrics[0]
        fit_op_not_zero = model.metrics[1]
        loss_op = model.metrics[2]

        sum_op = tf.summary.merge_all()
        sum_writer = tf.summary.FileWriter(self._env.summary_path, sess.graph)

        sess.run(init_op)
        norm_input_data = sess.run(model.before_train[3])
        #if norm_input_data == 0:
        #    fit_op = fit_op_zero
        #else:
        #    fit_op = fit_op_not_zero



        print('CP model initial finish')

        for step in range(steps):
            self._factors, self._lambdas, self._full_tensor, loss_v, fitness = sess.run([A, lambda_op, full_op, loss_op, fit_op_not_zero])
            if step + 1 == steps:
                sum_writer.add_summary(sess.run(sum_op), step)
                print('step=%d, RMSE=%f, fit=%.10f' % (step + 1, loss_v, fitness))

            elif args.verbose or step == 0 or step % args.validation_internal == 0:
                sum_writer.add_summary(sess.run(sum_op), step)
                print('step=%d, RMSE=%.10f, fit=%.10f' % (step + 1, loss_v, fitness))
            print(self._lambdas)
            #for matrix in self._factors:
            #    print(matrix)
            #    print('')

        print('CP model train finish, with RMSE = %.10f, fit=%.10f' % (loss_v, fitness))
        self._is_train_finish = True

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
        self._factors = None
        self._lambdas = None
        self._is_train_finish = False
        self._args = None
        self._init_op = None
        self._norm_input_data = None
        self._lambda_op = None
        self._full_op = None
        self._factor_update_op = None
        self._fit_op_zero = None
        self._fit_op_not_zero = None
        self._loss_op = None

    def predict(self, *key):
        if not self._full_tensor:
            raise TensorErr('improper stage to call predict before the model is trained')
        return self._full_tensor.item(key)

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

        with tf.name_scope('random-initial') as scope:
            A = [tf.Variable(rand(shape[ii], args.rank), name='A-%d' % ii) for ii in range(order)]
        with tf.name_scope('unfold-all-mode') as scope:
            mats = [ops.unfold(input_data, mode) for mode in range(order)]
            assign_op = [None for _ in range(order)]

        for mode in range(order):
            if mode != 0:
                with tf.control_dependencies([assign_op[mode - 1]]):
                    AtA = [tf.matmul(A[ii], A[ii], transpose_a=True, name='AtA-%d-%d' % (mode, ii)) for ii in
                           range(order)]
                    XA = tf.matmul(mats[mode], ops.khatri(A, mode, True), name='XA-%d' % mode)
            else:
                AtA = [tf.matmul(A[ii], A[ii], transpose_a=True, name='AtA-%d-%d' % (mode, ii)) for ii in range(order)]
                XA = tf.matmul(mats[mode], ops.khatri(A, mode, True), name='XA-%d' % mode)

            V = ops.hadamard(AtA, skip_matrices_index=mode)
            non_norm_A = tf.matmul(XA, tf.py_func(np.linalg.pinv, [V], tf.float64, name='pinvV-%d' % mode),
                                   name='XApinvV-%d' % mode)
            with tf.name_scope('max-norm-%d' % mode) as scope:
                lambda_op = tf.reduce_max(tf.reshape(non_norm_A, shape=(shape[mode], args.rank)), axis=0)
                assign_op[mode] = A[mode].assign(tf.div(non_norm_A, lambda_op))

        with tf.name_scope('full-tensor') as scope:
            P = KTensor(assign_op, lambda_op)
            full_op = P.extract()

        with tf.name_scope('loss') as scope:
            loss_op = rmse_ignore_zero(input_data, full_op)

        """
        if \\left \\| X - X_{real}  \\right \\|_F \\neq
        fitness = 1 - \\frac{\\left \\| X - X_{real}  \\right \\|_F}{\\left \\| X  \\right \\|_F}
        """
        with tf.name_scope('fitness') as scope:
            norm_input_data = tf.norm(input_data)
            fit_op_not_zero = 1 - tf.sqrt(
                tf.square(norm_input_data) + tf.square(tf.norm(full_op)) - 2 * ops.inner(input_data,
                                                                                         full_op)) / norm_input_data
            fit_op_zero = tf.square(tf.norm(full_op)) - 2 * ops.inner(input_data, full_op)

        tf.summary.scalar('loss', loss_op)
        tf.summary.scalar('fitness', fit_op_not_zero)
        tf.summary.scalar('fitness_0', fit_op_zero)

        init_op = tf.global_variables_initializer()

        self._args = args
        self._init_op = init_op
        self._norm_input_data = norm_input_data
        self._lambda_op = lambda_op
        self._full_op = full_op
        self._factor_update_op = assign_op
        self._fit_op_zero = fit_op_zero
        self._fit_op_not_zero = fit_op_not_zero
        self._loss_op = loss_op

    def train(self, steps):
        self._is_train_finish = False

        sess = self._env.sess
        args = self._args

        init_op = self._init_op
        norm_input_data = self._norm_input_data

        lambda_op = self._lambda_op
        full_op = self._full_op
        factor_update_op = self._factor_update_op

        # if the l2-norm of input tensor is zero, then use fit_op_zero
        # if not, then use fit_op_not_zero
        fit_op_zero = self._fit_op_zero
        fit_op_not_zero = self._fit_op_not_zero
        loss_op = self._loss_op

        sum_op = tf.summary.merge_all()
        sum_writer = tf.summary.FileWriter(self._env.summary_path, sess.graph)

        sess.run(init_op)
        print('CP model initial finish')

        if sess.run(norm_input_data) == 0:
            fit_op = fit_op_zero
        else:
            fit_op = fit_op_not_zero

        for step in range(1, steps + 1):
            if (step == steps) or (args.verbose) or (step == 1) or (step % args.validation_internal == 0):
                self._factors, self._lambdas, self._full_tensor, loss_v, fitness, sum_msg = sess.run(
                    [factor_update_op, lambda_op, full_op, loss_op, fit_op, sum_op])
                sum_writer.add_summary(sum_msg, step)
                print('step=%d, RMSE=%f, fit=%.10f' % (step, loss_v, fitness))
            else:
                self._factors, self._lambdas = sess.run([factor_update_op, lambda_op])

        print('CP model train finish, with RMSE = %.10f, fit=%.10f' % (loss_v, fitness))
        self._is_train_finish = True

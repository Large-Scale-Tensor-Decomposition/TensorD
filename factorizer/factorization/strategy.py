# Created by ay27 at 17/2/23
import tensorflow as tf
import factorizer.loss_template as ltp
import factorizer.base.type as type


class Strategy(object):
    def prepare(self, cluster):
        raise NotImplementedError

    def update(self, sess, feed_data):
        raise NotImplementedError

    def sync(self):
        raise NotImplementedError


class PPTTF(Strategy):
    def __init__(self, task_cnt, task_index, order, lamb, tao, rho):
        self.cluster = None
        self.task_cnt = task_cnt
        self.task_index = task_index
        self.order = order

        self.lamb = lamb
        self.tao = tao
        self.rho = rho

        self.supervisor = None
        self.global_step = None
        self.train_op = None

    def prepare(self, cluster):
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % self.task_index,
                cluster=cluster)):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

            X = tf.placeholder(tf.float64)
            A = tf.get_variable("A", dtype=tf.float64)
            B = tf.get_variable("B", dtype=tf.float64)
            C = tf.get_variable("C", dtype=tf.float64)

            ktensor = type.KTensor([A, B, C])

            self.loss_op = ltp.l2(X, ktensor.extract())

            grad = tf.gradients(self.loss_op, [A, B, C])

            new_A = A - self.tao * grad[0]
            new_B = B - self.tao * grad[1]
            new_C = C - self.tao * grad[2]

            assign_op1 = tf.assign(A, new_A)
            assign_op2 = tf.assign(B, new_B)
            assign_op3 = tf.assign(C, new_C)

            self.train_op = tf.group(assign_op1, assign_op2, assign_op3)

            saver = tf.train.Saver()
            tf.summary.histogram('loss', self.loss_op)
            summary_op = tf.summary.merge_all()
            init_op = tf.global_variables_initializer()

        self.supervisor = tf.train.Supervisor(is_chief=(self.task_index == 0),
                                              init_op=init_op,
                                              summary_op=summary_op,
                                              saver=saver,
                                              global_step=self.global_step,
                                              save_model_secs=60)

    def update(self, sess, feed_data):
        loss_v, _, step = sess.run([self.loss_op, self.train_op, self.global_step])
        return loss_v

    def sync(self):
        # TODO: how to sync data A,B,C to all workers?
        pass

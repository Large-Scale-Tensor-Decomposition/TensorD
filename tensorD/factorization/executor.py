# Created by ay27 at 17/2/22
import threading

import tensorflow as tf
from tensorD.base.logger import create_logger
from tensorD.base.barrier import Barrier
logger = create_logger()


def default_config():
    optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)
    config = tf.ConfigProto(
        graph_options=tf.GraphOptions(optimizer_options=optimizer_options))
    config.log_device_placement = False
    config.allow_soft_placement = False
    return config


class Executor(object):
    def __init__(self, ps_hosts, worker_hosts, role, task_index, data_provider, strategy=None, steps=100):
        """

        Parameters
        ----------
        ps_hosts
        worker_hosts
        role
        task_index
        data_provider
        strategy
        steps

        Returns
        -------

        """
        logger.debug('role=%s,type is %s' % (role, type(role)))
        if role != 'ps' and role != 'worker':
            raise ValueError("Role must be 'ps' or 'worker'")

        self.cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
        self.server = tf.train.Server(self.cluster, job_name=role, task_index=task_index)
        self.provider = data_provider
        self.strategy = strategy
        self.workers = worker_hosts
        self.role = role
        self.task_index = task_index

        self.steps = steps

    def train(self):
        self.barrier = Barrier(self.task_index)
        self.strategy.create_graph(self.cluster)
        if self.role == "ps":
            self.server.join()
        self.run()
        # [threading.Thread(target=self.run()) for _ in range(self.strategy.task_cnt)]

    def run(self):
        # with self.strategy.supervisor.managed_session(self.server.target) as sess:
        sess = tf.Session('grpc://' + self.workers[self.task_index])
        with sess:
            sess.run(self.strategy.init_op)
            self.barrier(sess)
            logger.error('run step, worker %d' % self.task_index)
            for step in range(self.steps):
                # for batch in self.provider.next_batch(2):
                logger.debug('fetch batch success')
                self.strategy.train(sess, feed_data=self.provider.tensor)
                self.barrier(sess)
                self.strategy.sync(sess, feed_data=self.provider.tensor)

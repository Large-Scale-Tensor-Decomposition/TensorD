# Created by ay27 at 17/2/22
import tensorflow as tf


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
        if role != 'ps' or role != 'worker':
            raise ValueError("Role must be 'ps' or 'worker'")

        self.cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
        self.server = tf.train.Server(self.cluster, job_name=role, task_index=task_index)
        self.provider = data_provider
        self.strategy = strategy
        self.role = role
        self.task_index = task_index

        self.steps = steps

    def train(self):
        if self.role == 'ps':
            self.server.join()
        else:
            self.strategy.create_graph(self.cluster)
            with self.strategy.supervisor.managed_session(self.server) as sess:
                for step in range(self.steps):
                    for batch in self.provider:
                        self.strategy.train(feed_data=batch)

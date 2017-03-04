# Created by ay27 at 17/2/24
import tensorflow as tf
import numpy as np
import time
from factorizer.factorization.executor import Executor
from factorizer.dataproc.provider import OrdProvider
from factorizer.dataproc.reader import TensorReader
from factorizer.factorization.strategy import PPTTF

"""
run:
python train.py --ps_hosts=172.17.0.3:2222 --worker_hosts=172.17.0.4:2223 --job_name=worker
python train.py --ps_hosts=172.17.0.3:2222 --worker_hosts=172.17.0.4:2223 --job_name=ps
"""

# Define parameters
FLAGS = tf.app.flags.FLAGS
# tf.app.flags.DEFINE_float('learning_rate', 0.004, 'Initial learning rate.')
# tf.app.flags.DEFINE_integer('steps_to_validate', 100,
#                             'Steps to validate and print loss')

# For distributed
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

tensor_data_file = '../data/tmp'

# Hyperparameters
# learning_rate = FLAGS.learning_rate
# steps_to_validate = FLAGS.steps_to_validate


def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    task_index = FLAGS.task_index
    task_cnt = len(worker_hosts)

    strategy = PPTTF(task_cnt, task_index, 3, 0.002, 0.002, 0.01)
    data_provider = OrdProvider(TensorReader(tensor_data_file), 3, task_cnt, task_index, 20, sparse=True)
    executor = Executor(ps_hosts, worker_hosts, FLAGS.job_name, FLAGS.task_index, data_provider, strategy, steps=200)

    executor.train()

if __name__ == "__main__":
    tf.app.run()

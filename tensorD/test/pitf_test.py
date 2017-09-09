from tensorD.factorization.env import Environment
from tensorD.factorization.pitf import PITF
from tensorD.factorization.tucker import *
from tensorD.dataproc.provider import Provider
#from tensorD.dataproc.reader import TensorReader
import tensorflow as tf
import numpy as np


if __name__ == '__main__':
    data_provider = Provider()
    data_provider.full_tensor = lambda: tf.constant(np.random.rand(50, 50, 8)*10, dtype=tf.float32)
    pitf_env = Environment(data_provider, summary_path='/tmp/tensord')
    pitf = PITF(pitf_env)
    args = PITF.PITF_Args(rank=5, delt=0.8, tao=12, sample_num=100, validation_internal=1)
    pitf.build_model(args)
    pitf.train(steps=500)

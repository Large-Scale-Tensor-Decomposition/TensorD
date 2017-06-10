# Created by ay27 at 17/6/6
from tensorD.factorization.env import Environment
from tensorD.factorization.cp import CP_ALS
from tensorD.factorization.tucker import *
from tensorD.dataproc.provider import Provider
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    data_provider = Provider()
    data_provider.full_tensor = lambda: tf.constant(np.arange(24).reshape(3, 4, 2), dtype=tf.float64)
    env = Environment(data_provider, summary_path='/tmp/tensord')
    # cp = CP_ALS(env)
    # args = CP_ALS.CP_Args(rank=2, validation_internal=1000)
    # cp.build_model(args)
    # cp.train(10000)
    # print(cp.full())

    # tucker = HOSVD(env)
    # args = HOSVD.HOSVD_Args([3,4,2])
    # tucker.build_model(args)
    # tucker.train(1000)
    # print(tucker.full())

    tucker = HOOI(env)
    args = HOOI.HOOI_Args([3,4,2], verbose=True)
    tucker.build_model(args)
    tucker.train(1000)
    print(tucker.full())

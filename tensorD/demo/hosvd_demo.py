from tensorD.factorization.env import Environment
from tensorD.dataproc.provider import Provider
from tensorD.factorization.tucker import HOSVD
from tensorD.demo.DataGenerator import *

if __name__ == '__main__':
    print('=========Train=========')
    X = synthetic_data_tucker([943, 1682, 31], [10, 10, 10])
    data_provider = Provider()
    data_provider.full_tensor = lambda: X
    env = Environment(data_provider, summary_path='/tmp/hosvd_demo')
    hosvd = HOSVD(env)
    args = HOSVD.HOSVD_Args(ranks=[20, 20, 20])
    hosvd.build_model(args)
    hosvd.train()
    factor_matrices = hosvd.factors
    core_tensor = hosvd.core
    print('Train ends.\n\n\n')

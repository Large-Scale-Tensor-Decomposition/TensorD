from tensorD.dataproc.reader import TensorReader
import tensorflow as tf
from tensorD.factorization.env import Environment
from tensorD.dataproc.provider import Provider
from tensorD.factorization.cp import CP_ALS
from tensorD.factorization.ncp import NCP_BCU
from tensorD.factorization.ntucker import NTUCKER_BCU
from tensorD.factorization.tucker import HOOI
from tensorD.demo.DataGenerator import *

if __name__=='__main__':
    full_shape = [943, 1682, 31]
    base = TensorReader('/root/tensorD_f/data_out_tmp/u1.base.csv')
    base.read(full_shape=full_shape)
    with tf.Session() as sess:
        rating_tensor = sess.run(base.full_data)
    data_provider = Provider()
    data_provider.full_tensor = lambda : rating_tensor
    # env = Environment(data_provider,summary_path='/tmp/cp_ml')
    # env = Environment(data_provider, summary_path='/tmp/ncp_ml')
    # env = Environment(data_provider, summary_path='/tmp/ntucker_ml')
    env = Environment(data_provider, summary_path='/tmp/tucker_ml')
    # ntucker = NTUCKER_BCU(env)
    # cp = CP_ALS(env)
    # ncp = NCP_BCU(env)
    hooi = HOOI(env)
    # args = CP_ALS.CP_Args(rank=20,validation_internal=1)
    # args = NCP_BCU.NCP_Args(rank=20,validation_internal=1)
    # args = NTUCKER_BCU.NTUCKER_Args(ranks=[20,20,20],validation_internal=20)
    args = HOOI.HOOI_Args(ranks=[20,20,20],validation_internal=1)
    # ntucker.build_model(args)
    # cp.build_model(args)
    # ncp.build_model(args)
    hooi.build_model(args)
    # loss_hist = cp.train(100)
    # loss_hist = ncp.train(100)
    # loss_hist = ntucker.train(2000)
    loss_hist = hooi.train(10)
    # out_path = '/root/tensorD_f/data_out_tmp/python_out/cp_ml_20.txt'
    # out_path = '/root/tensorD_f/data_out_tmp/python_out/ncp_ml_20.txt'
    # out_path = '/root/tensorD_f/data_out_tmp/python_out/ntucker_ml_20.txt'
    out_path = '/root/tensorD_f/data_out_tmp/python_out/tucker_ml_20.txt'
    with open(out_path, 'w') as out:
        for loss in loss_hist:
            out.write('%.6f\n' % loss)


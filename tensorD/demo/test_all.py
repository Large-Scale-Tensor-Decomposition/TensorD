#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/9 PM9:47
# @Author  : Shiloh Leung
# @Site    : 
# @File    : test_all.py
# @Software: PyCharm Community Edition

from tensorD.factorization.env import Environment
from tensorD.dataproc.provider import Provider
from tensorD.factorization.ntucker import NTUCKER_BCU
from tensorD.factorization.tucker import HOOI
from tensorD.factorization.cp import CP_ALS
from tensorD.factorization.ncp import NCP_BCU
from tensorD.demo.DataGenerator import *
import time

if __name__ == '__main__':
    # # cp test
    # for ii in range(1, 6):
    #     I = 30 * ii
    #     R = 10 + ii
    #     X = synthetic_data_nonneg([I, I, I], R)
    #     data_provider = Provider()
    #     data_provider.full_tensor = lambda: X
    #     env = Environment(data_provider, summary_path='/tmp/cp_demo_' + str(I))
    #     cp = CP_ALS(env)
    #     args = CP_ALS.CP_Args(rank=R, validation_internal=5)
    #     cp.build_model(args)
    #     print('CP with %dx%dx%d, R=%d' % (I, I, I, R))
    #     start = time.time()
    #     cp.train(100*ii)
    #     print('Train ends in %.5f\n\n' % time.time() - start)

    # ncp test
    # for ii in range(1, 2):
    #     I = 30*6
    #     R = 20
    #     X = synthetic_data_nonneg([I, I, I], R)
    #     data_provider = Provider()
    #     data_provider.full_tensor = lambda: X
    #     env = Environment(data_provider, summary_path='/tmp/ncp_demo_' + str(I))
    #     ncp = NCP_BCU(env)
    #     args = NCP_BCU.NCP_Args(rank=R, validation_internal=5)
    #     ncp.build_model(args)
    #     print('NCP with %dx%dx%d, R=%d' % (I, I, I, R))
    #     start = time.time()
    #     ncp.train(ii*100)
    #     print('Train ends in %.5f\n\n' % (time.time() - start))

    # ntucker test

    N1 = 40
    N2 = 40
    N3 = 40
    q = 10
    R = 10
    X = synthetic_data_nonneg([N1, N2, N3], q)
    data_provider = Provider()
    data_provider.full_tensor = lambda: X
    env = Environment(data_provider, summary_path='/tmp/ntucker_demo_' + str(N1))
    ntucker = NTUCKER_BCU(env)
    args = NTUCKER_BCU.NTUCKER_Args(ranks=[R, R, R], validation_internal=1)
    ntucker.build_model(args)
    print('NTucker with %dx%dx%d, R=%d' % (N1, N2, N3, q))
    loss_hist = ntucker.train(500)
    out_file = '/root/tensorD_f/tensorD_OUT/ntucker_N1_N2_N3_q_R_' + str(N1) + '_' + str(N2) + '_' + str(N3) + '_' + str(q) + '_' + str(R) + '1'
    with open(out_file, 'w') as out:
        for loss in loss_hist:
            out.write('%.5f\n' % loss)

    # # tucker test
    # for ii in range(1, 6):
    #     I = 30 * ii
    #     R = 10 + ii
    #     X = synthetic_data_nonneg([I, I, I], R)
    #     data_provider = Provider()
    #     data_provider.full_tensor = lambda: X
    #     env = Environment(data_provider, summary_path='/tmp/tucker_demo_' + str(I))
    #     hooi = HOOI(env)
    #     args = HOOI.HOOI_Args(ranks=[R, R, R], validation_internal=3)
    #     hooi.build_model(args)
    #     print('Tucker with %dx%dx%d, R=%d' % (I, I, I, R))
    #     start = time.time()
    #     hooi.train(50)
    #     print('Train ends in %.5f\n\n' % time.time() - start)

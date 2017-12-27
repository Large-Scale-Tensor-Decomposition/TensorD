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
    # cp test
    # N1 = 160
    # N2 = 160
    # N3 = 160
    # R = 10
    # X = synthetic_data_cp([N1, N2, N3], R)
    # data_provider = Provider()
    # data_provider.full_tensor = lambda: X
    # env = Environment(data_provider, summary_path='/tmp/cp_demo_' + str(N1))
    # cp = CP_ALS(env)
    # args = CP_ALS.CP_Args(rank=R, validation_internal=1)
    # cp.build_model(args)
    # print('CP with %dx%dx%d, R=%d' % (N1, N2, N3, R))
    # loss_hist = cp.train(25)
    # # out_path = '/root/tensorD_f/data_out_tmp/python_out/cp_N1_N2_N3_R_' + str(N1) + '_' + str(N2) + '_' + str(
    # #     N3) + '_' + str(R) + '_1.txt'
    # out_path = '/root/tensorD_f/data_out_tmp/python_out/cp_160_10_20.txt'
    # with open(out_path, 'w') as out:
    #     for loss in loss_hist:
    #         out.write('%.6f\n' % loss)

    # ncp test
    # N1 = 160
    # N2 = 160
    # N3 = 160
    # R = 10
    # X = synthetic_data_cp([N1, N2, N3], R)
    # data_provider = Provider()
    # data_provider.full_tensor = lambda: X
    # env = Environment(data_provider, summary_path='/tmp/ncp_demo_' + str(N1))
    # ncp = NCP_BCU(env)
    # args = NCP_BCU.NCP_Args(rank=R, validation_internal=1)
    # ncp.build_model(args)
    # print('NCP with %dx%dx%d, R=%d' % (N1, N2, N3, R))
    # loss_hist = ncp.train(100)
    # # out_path = '/root/tensorD_f/data_out_tmp/python_out/ncp_N1_N2_N3_R_' + str(N1) + '_' + str(N2) + '_' + str(
    # #     N3) + '_' + str(R) + '_2.txt'
    # out_path = '/root/tensorD_f/data_out_tmp/python_out/ncp_160_10_20.txt'
    # with open(out_path, 'w') as out:
    #     for loss in loss_hist:
    #         out.write('%.6f\n' % loss)

    # # ntucker test
    N1 = 160
    N2 = 160
    N3 = 160
    R = 10
    X = synthetic_data_tucker([N1, N2, N3], [R, R, R])
    data_provider = Provider()
    data_provider.full_tensor = lambda: X
    env = Environment(data_provider, summary_path='/tmp/ntucker_demo_' + str(N1))
    ntucker = NTUCKER_BCU(env)
    args = NTUCKER_BCU.NTUCKER_Args(ranks=[R, R, R], validation_internal=200, tol=1.0e-4)
    ntucker.build_model(args)
    print('NTucker with %dx%dx%d, R=%d' % (N1, N2, N3, R))
    loss_hist = ntucker.train(6000)
    # out_path = '/root/tensorD_f/data_out_tmp/python_out/ntucker_N1_N2_N3_R_' + str(N1) + '_' + str(N2) + '_' + str(
    #     N3) + '_' + str(R) + '_1.txt'
    out_path = '/root/tensorD_f/data_out_tmp/python_out/ntucker_160_10_20.txt'
    with open(out_path, 'w') as out:
        for loss in loss_hist:
            out.write('%.6f\n' % loss)

    # tucker test
    # N1 = 160
    # N2 = 160
    # N3 = 160
    # R = 10
    # X = synthetic_data_tucker([N1, N2, N3], [R, R, R])
    # data_provider = Provider()
    # data_provider.full_tensor = lambda: X
    # env = Environment(data_provider, summary_path='/tmp/tucker_demo_' + str(N1))
    # hooi = HOOI(env)
    # args = HOOI.HOOI_Args(ranks=[R, R, R], validation_internal=1)
    # hooi.build_model(args)
    # print('Tucker with %dx%dx%d, R=%d' % (N1, N2, N3, R))
    # loss_hist = hooi.train(2)
    # out_path = '/root/tensorD_f/data_out_tmp/python_out/tucker_N1_N2_N3_R_' + str(N1) + '_' + str(N2) + '_' + str(
    #     N3) + '_' + str(R) + '_1.txt'
    # with open(out_path, 'w') as out:
    #     for loss in loss_hist:
    #         out.write('%.6f\n' % loss)

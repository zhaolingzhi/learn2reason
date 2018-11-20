# !/usr/bin/env
# coding=utf-8

import pickle
import logging

import numpy as np
import time
from keras.utils import to_categorical

from models.prop_logics.model.DNN import DNN
from models.prop_logics.model.DQN import DQN
from models.prop_logics.LOG import LOG


vmf = "good_params_vm.pkl"
smf = "sm_model.h5"
curriculum_sm = "curriculum_sm.pkl"

docs = {}
docs['0'] = 1
docs['1'] = 2
docs['and'] = 3
docs['or'] = 4
docs['not'] = 5
docs['('] = 6
docs[')'] = 7

n0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0]  # place holder
n1 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0]  # 0
n2 = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0]  # 1
n3 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0]  # and
n4 = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0]  # or
n5 = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0]  # ~
n6 = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0]  # (
n7 = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0]  # )
one_hots = [n0, n1, n2, n3, n4, n5, n6, n7]
leaves = ['0', '1', 'and', 'or', 'not', '(', ')',
          '0 and', '0 or', '0 )', '1 and', '1 or', '1 )',
          'and 0', 'and 1', 'and not', 'and (',
          'or 0', 'or 1', 'or not', 'or (',
          'not 0', 'not 1', 'not not', 'not (',
          '( 0', '( 1', '( not', '( (',
          ') and', ') or', ') )', 'error!',
          '( 0 and', '( 0 or', '( 1 and', '( 1 or']
start_epoches = 10
batch_size = 400

_log=LOG(file="sm.log",name="SM")


def to_one_hot(value):
    max = np.amax(value)
    value[value < max] = 0
    value[value >= max] = 1
    return value


def main():
    with open(curriculum_sm, 'rb') as f:
        tx, ty, cs, mises = pickle.load(f)
        # tx[0]:[6,6,0,7,7]
        # ty[0]:0
        # cs[0]:"((0))"
        # mises[0]:[1,1,0,0]
    vm = DNN(input=74, output=37)
    sm = DQN(state_size=74, action_size=5)
    vm.loadweight(vmf)
    for i in range(len(tx)):
        for j in range(len(tx[i])):
            tx[i][j] = one_hots[tx[i][j]]
    ty = to_categorical(np.asarray(ty), num_classes=37)

    ein_list = []
    eout_list = []
    epri_list = []
    count = 0
    start_time=time.time()
    save_wait = 5

    while True:
        count += 1
        correct = 0
        for i in range(len(tx)):
            etx = tx[i]  #每个x
            ety = ty[i]  #每个y
            ecs = cs[i]
            emises = mises[i]
            etx_copy = np.copy(etx)
            ein_l = []   #模型输入数据
            eout_l = []  #VM输出数据
            epri_l = []  #sm优先级数据
            eprv_l = []  #sm优先数
            for j in range(len(etx) - 1):
                ein = []
                epri = []
                eout = []
                eprv = []
                for k in range(etx_copy.shape[0] - 1):
                    sub_etx = np.concatenate([etx_copy[k], etx_copy[k + 1]])
                    sub_etx = to_one_hot(sub_etx)
                    sub_etx = np.asarray([sub_etx])
                    sub_out = vm.predict(sub_etx)
                    sub_prv = sm.predict(sub_etx)
                    sub_pri = np.argmax(sub_prv)

                    ein.append(sub_etx)
                    eout.append(sub_out[0])
                    epri.append(sub_pri)
                    eprv.append(sub_prv)

                ein_l.append(ein)  # state and next state
                eout_l.append(eout)  # used in reward
                epri_l.append(epri)  # action
                eprv_l.append(eprv)
                pri_idx = np.argmax(epri)
                eout = np.asarray(eout)
                etx_copy = np.concatenate([etx_copy[:pri_idx], eout[pri_idx:pri_idx + 1],
                                           etx_copy[pri_idx + 2:]], axis=0)

            if np.argmax(etx_copy) == np.argmax(ety):
                correct += 1
            # ein_list.append(ein_l)
            # eout_list.append(eout_l)
            # epri_list.append(epri_l)
            result_r = etx_copy[0, np.argmax(ety)]
            sm.remember(ein_l, eout_l, epri_l, result_r, eprv_l)
            # print("index:%d"%i)
        loss = 0
        if count > start_epoches:
            loss = sm.replay(batch_size)
        used_time=time.time()-start_time
        _log.getlogger().info("epoche:%d,full=%d,correct=%d,loss=%f,time=%f" % (count, len(tx), correct,loss,used_time))

        if len(tx)==correct:
            save_wait -= 1
        if save_wait == 0:
            return



if __name__ == "__main__":
    main()

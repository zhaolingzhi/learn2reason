# !/usr/bin/env
# coding=utf-8

import pickle

import numpy as np
from keras.utils import to_categorical

from models.prop_logics.model.DNN import DNN
from models.prop_logics.model.DQN import DQN

vmf = "vm_model.h5"
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
batch_size = 128


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
    sm = DQN(state_size=74, action_size=5,test=True)
    sm.load()
    vm.loadweight(vmf)
    for i in range(len(tx)):
        for j in range(len(tx[i])):
            tx[i][j] = one_hots[tx[i][j]]
    ty = to_categorical(np.asarray(ty), num_classes=37)

    correct = 0

    for i in range(len(tx)):
        etx = tx[i]
        ety = ty[i]
        ecs = cs[i]
        emises = mises[i]
        etx_copy = np.copy(etx)
        ein_l = []
        eout_l = []
        epri_l = []
        for j in range(len(etx) - 1):
            ein = []
            epri = []
            eout = []
            for k in range(etx_copy.shape[0] - 1):
                sub_etx = np.concatenate([etx_copy[k], etx_copy[k + 1]])
                sub_etx = to_one_hot(sub_etx)
                sub_etx = np.asarray([sub_etx])
                sub_out = vm.predict(sub_etx)
                sub_pri = sm.act(sub_etx)

                ein.append(sub_etx)
                eout.append(sub_out[0])
                epri.append(sub_pri)
            ein_l.append(ein)  # state and next state
            eout_l.append(eout)  # used in reward
            epri_l.append(epri)  # action
            pri_idx = np.argmax(epri)
            eout = np.asarray(eout)
            etx_copy = np.concatenate([etx_copy[:pri_idx], eout[pri_idx:pri_idx + 1],
                                       etx_copy[pri_idx + 2:]], axis=0)

        if np.argmax(etx_copy) == np.argmax(ety):
            correct += 1

            # print("index:%d"%i)
    print("full=%d,correct=%d" % (len(tx), correct))


if __name__ == "__main__":
    main()

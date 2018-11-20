# !/usr/bin/env python
import sys
sys.path.append('..')
import os
import random
import argparse
import numpy as np
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]=""
import scipy
import scipy.misc
import pickle
import time

np.set_printoptions(precision=3,linewidth=200,threshold=10000000, suppress=True)

maxlen = 800
nsymbols = 17
ncombs = 40
v_epsilon = 1e-9
n_as = 4
lh = 1

S_Acts = [0, 1, 2, 3]

prq_a = 0.5
# prq_a = 0.8
prq_e = 1e-6
lr1 = 1e-3
# lr1 = 0.01
init_v = 10*lr1
# lr1 = 1
# init_v = 0.5
env_dim = ncombs*2
rg = np.random.RandomState(seed=234)

one_hots = []
for i in range(nsymbols):
    t = np.zeros(nsymbols)
    t[i] = 1
    one_hots.append(t)

def one_hot_preprocess(xs):
    new_xs = []
    for x in xs:
        new_xs.append(np.concatenate([one_hots[x], [0]*(ncombs-nsymbols)]))
    for i in range(maxlen - len(xs)):
        new_xs.append([0]*ncombs)
    return new_xs


def e_predict(e_in, e_param):
    W_n1, b_n1, W_n2, b_n2 = e_param
    e_out = np.dot(e_in, W_n1) + b_n1
    e_out = np.maximum(e_out, 0)
    e_out = np.dot(e_out, W_n2) + b_n2
    return e_out

def s_predict(s_in, s_param):
    W_s1, b_s1, W_s2, b_s2 = s_param
    score = np.dot(s_in, W_s1) + b_s1
    score = np.maximum(score, 0)
    score = np.dot(score, W_s2) + b_s2
    # score = np.maximum(score, 0)
    return score

def compute_max_idxs(e_param, s_param, expr, lens, target):
    W_n1, b_n1, W_n2, b_n2 = e_param

    max_idxs = np.zeros((maxlen - 1), dtype=np.int32)
    loop_idx = 0
    lens = lens[0]
    exprs = []
    score_acts = []
    e_ins = []
    selected_e_ins = []
    while expr.shape[0] > 1 + maxlen - lens:
        nexprs = []
        scores = []
        score_act = []
        e_in_list = []
        for i in range(0, lens - loop_idx - 1):
            e_in = np.concatenate([expr[i], expr[i+1]])

            c = np.copy(e_in)
            e_in[c < 0.1] = 0 
            # e_in[c < 0.7] = 0 
            e_in[c > 0.9] = 1 
            # print('e in', e_in)
            
            e_in_list.append(e_in)
            e_out = np.dot(e_in, W_n1) + b_n1
            e_out = np.maximum(e_out, 0)
            e_out = np.dot(e_out, W_n2) + b_n2
            # SL Expr
            nexprs.append(e_out)

            if lh == 1:
                e_in_t = e_in
            else:
                e_in_t = np.concatenate([np.asarray(selected_e_ins[1-lh:]).flatten(), e_in])
                e_in_t = np.concatenate([np.asarray([0]*(np.maximum(0, lh*2*ncombs - len(e_in_t)))), e_in_t])
            score = s_predict(e_in_t, s_param)

            # print('score', score)

            score_act.append(np.argmax(score))
            scores.append(S_Acts[np.argmax(score)])

        e_ins.append(np.asarray(e_in_list))
        scores = np.asarray(scores)

        # max_idx = np.argmax(scores)
        max_idx = np.where(scores == np.amax(scores))[0][-1]
        score_acts.append(score_act)
        selected_e_ins.append(e_in_list[max_idx])

        # print(scores)

        # print(e_in_list[max_idx])
        # print(nexprs[max_idx:max_idx+1])

        expr = np.concatenate([expr[:max_idx], nexprs[max_idx:max_idx+1], expr[max_idx+2:]], axis=0)

        max_idxs[loop_idx] = max_idx
        loop_idx += 1

    fexpr = expr[0]

    # r0 = np.argmax(fexpr) == target
    # r0 = fexpr[target] + float(np.argmax(fexpr) == target )
    r0 = fexpr[target]
    selected_e_ins = np.asarray(selected_e_ins)

    return max_idxs, fexpr, r0


if __name__ == '__main__':
    fname = './logic_t6.pkl'
    # fname = './ite_expr_1.pkl'
    inputs = pickle.load(open(fname))
    tx, ty, cs = inputs[:3]
    print(tx[0], ty[0], cs[0])
    batch_size = 400
    tbatch = 16
    start_epoch = 80

    # only train part
    part = 8000
    # part = 352
    
    tx = tx[:part]
    ty = ty[:part]
    cs = cs[:part]

    train_x = []
    train_y1 = []
    train_y2 = []
    len_xss = []

    for idx in range(len(tx)):
        x = tx[idx]
        len_xs = np.asarray([len(x)])
        xs = one_hot_preprocess(x)
        xs = np.asarray(xs)
        y = ty[idx]
        yt1 = np.zeros((1,ncombs), dtype=np.int32)
        yt1[0][y] = 1
        yt2 = np.asarray([y], dtype=np.int32)
        train_x.append(xs)
        train_y1.append(yt1)
        train_y2.append(yt2)
        len_xss.append(len_xs)
    train_x = np.asarray(train_x)
    train_y1 = np.asarray(train_y1)
    train_y2 = np.asarray(train_y2)
    len_xss = np.asarray(len_xss)


    # read good params
    params1 = pickle.load(open('./good02_params_bn1.pkl'))
    W_n1, b_n1, W_n2, b_n2 = params1
    e_param = [W_n1, b_n1, W_n2, b_n2]
    # e_param = [tW_n1, tb_n1, tW_n2, tb_n2]

    # params2 = pickle.load(open('./good_params_rl4_2.pkl'))
    params2 = pickle.load(open('./good_params_rl4_2.pkl'))
    s_param = params2[1]

    # s_param = [tW_s1, tb_s1, tW_s2, tb_s2]

    correct = 0
    tcorrect = 0
    mcorrect = 0
    full = 0
    sum_r = 0.0
    sum_loss = 0
    fail_cs = []
    for idx in range(len(tx)):
        # if idx != 14:
            # continue
        xs = train_x[idx]
        yt1 = train_y1[idx]
        yt2 = train_y2[idx]
        len_xs = len_xss[idx]

        ms, nexpr, r0= compute_max_idxs(e_param, s_param, xs, len_xs, yt2)

        sum_r += r0
        # mis = mises[idx]
        print('index', idx, 'reward', r0, cs[idx],  nexpr[:2], ms[:7])
        # if all(mis == ms[:len(mis)]):
        #     mcorrect += 1
        if np.argmax(nexpr) == ty[idx]:
            correct += 1
            if nexpr[ty[idx]] > 0.8:
                tcorrect += 1
            else:
                fail_cs.append(cs[idx])
        else:
            fail_cs.append(cs[idx])
        full += 1


    print('correct', correct, 'true correct', tcorrect, 'm correct' , mcorrect, 'full', full)
    
    for c in fail_cs:
        print(c)



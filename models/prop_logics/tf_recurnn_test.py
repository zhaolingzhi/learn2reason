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
docs = {}
docs['0'] = 1
docs['1'] = 2
docs['and'] = 3
docs['or'] = 4
docs['not'] = 5
docs['('] = 6
docs[')'] = 7

n0 = [0,0,0,0,0,0,0] # place holder
n1 = [1,0,0,0,0,0,0] # 0
n2 = [0,1,0,0,0,0,0] # 1
n3 = [0,0,1,0,0,0,0] # and
n4 = [0,0,0,1,0,0,0] # or
n5 = [0,0,0,0,1,0,0] # ~
n6 = [0,0,0,0,0,1,0] # (
n7 = [0,0,0,0,0,0,1] # )
one_hots = [n0,n1,n2,n3,n4,n5,n6,n7]
leaves = ['0', '1', 'and', 'or', 'not', '(', ')',\
        '0 and', '0 or', '0 )', '1 and', '1 or', '1 )',\
        'and 0', 'and 1', 'and not', 'and (',\
        'or 0', 'or 1', 'or not', 'or (',\
        'not 0', 'not 1', 'not not', 'not (',\
        '( 0', '( 1', '( not', '( (',\
        ') and', ') or', ') )', 'error!',\
        '( 0 and', '( 0 or', '( 1 and', '( 1 or']

nsymbols = 7
ncombs = 37
v_epsilon = 1e-9
n_ae = 12
n_as = 5
lh = 1

e_leaves = ['0', '1', '0 and', '0 or', '1 and', '1 or', '( 1', '( 0', '( 0 and', '( 0 or', '( 1 and', '( 1 or']
E_Acts = [np.zeros((ncombs,)) for i in range(len(e_leaves))]
for i in range(len(e_leaves)):
    E_Acts[i][leaves.index(e_leaves[i])] = 1

S_Acts = [0, 1, 2, 3, 4]

tW_n1 = np.zeros((ncombs*ncombs, 2*ncombs))
tb_n1 = np.zeros((ncombs*ncombs,))
tW_n2 = np.zeros((ncombs, ncombs*ncombs))
tb_n2 = np.zeros((ncombs,))

for i in range(ncombs):
    for j in range(ncombs):
        tW_n1[i*ncombs+j][i] = 1
        tW_n1[i*ncombs+j][j+ncombs] = 1

tb_n1 -= 1
tW_n2[0][7*ncombs+0] = 1
tW_n2[0][7*ncombs+1] = 1
tW_n2[0][10*ncombs+0] = 1
tW_n2[0][8*ncombs+0] = 1
tW_n2[0][25*ncombs+6] = 1
tW_n2[0][4*ncombs + 1] = 1
tW_n2[0][23*ncombs+0] = 1
tW_n2[1][11*ncombs+0] = 1
tW_n2[1][11*ncombs+1] = 1
tW_n2[1][10*ncombs+1] = 1
tW_n2[1][8*ncombs+1] = 1
tW_n2[1][26*ncombs+6] = 1
tW_n2[1][4*ncombs+0] = 1
tW_n2[1][23*ncombs+1] = 1
tW_n2[4][23*ncombs+4] = 1
tW_n2[5][5*ncombs+5] = 1
tW_n2[6][6*ncombs+6] = 1
tW_n2[7][0*ncombs+2] = 1
tW_n2[8][0*ncombs+3] = 1
tW_n2[9][0*ncombs+6] = 1
tW_n2[10][1*ncombs+2] = 1
tW_n2[11][1*ncombs+3] = 1
tW_n2[12][1*ncombs+6] = 1
tW_n2[23][4*ncombs+4] = 1
tW_n2[24][4*ncombs+5] = 1
tW_n2[25][5*ncombs+0] = 1
tW_n2[26][5*ncombs+1] = 1
tW_n2[33][25*ncombs+2] = 1
tW_n2[34][25*ncombs+3] = 1
tW_n2[35][26*ncombs+2] = 1
tW_n2[36][26*ncombs+3] = 1
tW_n2[25][33*ncombs+0]  = 1
tW_n2[25][33*ncombs+1]  = 1
tW_n2[25][35*ncombs+0]  = 1
tW_n2[26][35*ncombs+1]  = 1
tW_n2[25][34*ncombs+0]  = 1
tW_n2[26][34*ncombs+1]  =1 
tW_n2[26][36*ncombs+0]  =1 
tW_n2[26][36*ncombs+1]  =1 
tb_n2[32] = 0.5

tW_n2 = tW_n2.T
tW_n1 = tW_n1.T

tW_s1 = np.zeros((ncombs * ncombs, 2*ncombs))
tb_s1 = np.zeros((ncombs * ncombs, ))
tW_s2 = np.zeros((n_as, ncombs * ncombs))
tb_s2 = np.zeros((n_as, ))

for i in range(ncombs):
    for j in range(ncombs):
        tW_s1[i*ncombs+j][i] = 1
        tW_s1[i*ncombs+j][j+ncombs] = 1

tb_s1 -= 1

# ( 0 | ), ( 1 | )
tW_s2[4][25*ncombs+6] = 1
tW_s2[4][26*ncombs+6] = 1

# # not not
# tW_s2[3][4*ncombs+4] = 1

# # not not | 0,1
# tW_s2[3][23*ncombs+0] =1 
# tW_s2[3][23*ncombs+1] =1 

# not 0, not 1
tW_s2[4][4*ncombs+0] =1 
tW_s2[4][4*ncombs+1] =1 

# 0 and 0/1, 1 and 0/1
tW_s2[3][7*ncombs+0]  =1 
tW_s2[3][7*ncombs+1]  =1 
tW_s2[3][10*ncombs+0] =1 
tW_s2[3][10*ncombs+1] =1 

# 0/1 | and 
tW_s2[3][0*ncombs+2]  = 1
tW_s2[3][1*ncombs+2]  = 1

# 0 or 0/1, 1 or 0/1
tW_s2[1][8*ncombs+0]  =1 
tW_s2[1][8*ncombs+1]  =1 
tW_s2[1][11*ncombs+0] =1 
tW_s2[1][11*ncombs+1] =1 

# 0/1 | or 
tW_s2[1][0*ncombs+3]  = 1
tW_s2[1][1*ncombs+3]  = 1

# ( | 0/1
tW_s2[4][5*ncombs+0]  = 1
tW_s2[4][5*ncombs+1]  = 1

# ( 0/1 and 0/1
tW_s2[4][33*ncombs+0]  =1
tW_s2[4][33*ncombs+1]  =1
tW_s2[4][35*ncombs+0]  =1
tW_s2[4][35*ncombs+1]  =1

# ( 0/1 | and
tW_s2[4][25*ncombs+2]  =1
tW_s2[4][26*ncombs+2]  =1 

# ( 0/1  or 0/1
tW_s2[2][34*ncombs+0]  =1
tW_s2[2][34*ncombs+1]  =1
tW_s2[2][36*ncombs+0]  =1
tW_s2[2][36*ncombs+1]  =1

# ( 0/1 | or
tW_s2[2][25*ncombs+3]  =1
tW_s2[2][26*ncombs+3]  =1 

tb_s2[0] = 0.5

tW_s2 = tW_s2.T
tW_s1 = tW_s1.T

lr1 = 0.001
# lr1 = 0.01
init_v = 10*lr1
# lr1 = 1
# init_v = 0.5
prq_a = 0.5
prq_e = 1e-6
rg = np.random.RandomState(seed=234)

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
            e_in[c > 0.9] = 1 
            # print('e in', e_in)
            
            e_in_list.append(e_in)
            e_out = np.dot(e_in, W_n1) + b_n1
            e_out = np.maximum(e_out, 0)
            e_out = np.dot(e_out, W_n2) + b_n2
            # RL Expr
            # nexprs.append(E_Acts[np.argmax(e_out)])
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

        max_idx = np.argmax(scores)
        # max_idx = np.where(scores == np.amax(scores))[0][-1]
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
    fname = './logic_long.pkl'
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
    params1 = pickle.load(open('./good_params_vm.pkl'))
    W_n1, b_n1, W_n2, b_n2 = params1
    e_param = [W_n1, b_n1, W_n2, b_n2]

    params2 = pickle.load(open('./good_params_sm.pkl'))
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
        # if idx != 424:
        #     continue
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



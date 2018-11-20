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
from tf_recurnn_rlnn import Score_lh_NN
from SumTree import SumTree

np.set_printoptions(precision=3,linewidth=200,threshold=10000000, suppress=True)

maxlen = 30
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

def one_hot_preprocess(xs):
    new_xs = []
    for x in xs:
        new_xs.append(np.concatenate([one_hots[x], [0]*(ncombs-nsymbols)]))
    for i in range(maxlen - len(xs)):
        new_xs.append([0]*ncombs)
    return new_xs


class ExperienceReplay(object):
    def __init__(self, max_memory=100000, discount=.9):
        self.s_memory = SumTree(max_memory)
        self.discount = discount

    def remember(self, e_in_ts, score_acts, exprs_ts, max_idxs, r0, params, predict_model):
        r0 = r0*2-1  #dynamic priority   between (-1 1)

        # reward for score action
        for i in range(len(score_acts)):
            score_act = score_acts[i]
            for j in range(len(score_act)):
                r = 0
                # r = r0

                # if np.amax(exprs_ts[i][j]) < 0.9 and score_act[j] != 0:
                #     # r = -1
                #     r -= 0.3

                if np.amax(exprs_ts[i][j]) > 0.9 and score_act[j] != 0:
                    r += 0.1

                if i < len(score_acts) - 1:
                    state = ((e_in_ts[i][j], score_act[j], r, e_in_ts[i+1][max_idxs[i+1]]), 0) #not endding
                else:
                    r += r0   #the end
                    state = ((e_in_ts[i][j], score_act[j], r, e_in_ts[i][j]), 1)

                v = predict_model(np.asarray([state[0][0]]), params)[0]
                v = v[state[0][1]]  # get the the value of result in priority
                p = np.power(np.abs(r - v) + prq_e, prq_a)  #SumTree priority
                # print('p val', p)
                self.s_memory.add(p, state)

    def _get_batch(self, predict_model, params, o_params, nactions, memory, batch_size=200):
        inputs = np.zeros((batch_size, env_dim))
        targets = np.zeros((inputs.shape[0], nactions))
        for i, s in enumerate(np.random.rand(batch_size)*memory.total()):
            idx, val, data = memory.get(s)
            state_t, action_t, reward_t, state_tp1 = data[0]
            game_over = data[1]

            inputs[i:i+1] = np.asarray(state_t)
            targets[i] = predict_model(np.asarray([state_t]), params)[0] # according to double dqn not change
            if game_over == 1:  # if game_over is True
                targets[i, action_t] = reward_t
            else:
                # reward_t + gamma * max_a' Q(s', a')
                Q_sa_i = np.argmax(predict_model(np.asarray([state_tp1]), params)[0]) #double dqn
                Q_sa = predict_model(np.asarray([state_tp1]), o_params)[0][Q_sa_i]
                targets[i, action_t] = reward_t + self.discount * Q_sa
        return inputs, targets

    def get_s_batch(self, s_param, o_s_param, batch_size=200):
        return self._get_batch(s_predict, s_param, o_s_param, n_as, self.s_memory, batch_size)


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
    score = np.dot(score, W_s2) + b_s2   #compute the sm for this input
    return score

def compute_max_idxs(e_param, s_param, expr, lens, target, er, s_eps, isTrain=True):
    W_n1, b_n1, W_n2, b_n2 = e_param

    max_idxs = np.zeros((maxlen - 1), dtype=np.int32)
    loop_idx = 0
    lens = lens[0]
    exprs_ts = []
    score_acts = []
    e_in_ts = []
    selected_e_in_ts = []
    while expr.shape[0] > 1 + maxlen - lens:
        nexprs = []
        scores = []
        score_act = []
        e_in_t_list = []
        for i in range(0, lens - loop_idx - 1):
            e_in = np.concatenate([expr[i], expr[i+1]])

            # e_in = np.round(e_in)
            c = np.copy(e_in)
            e_in[c < 0.1] = 0 
            e_in[c > 0.9] = 1 
            
            e_out = np.dot(e_in, W_n1) + b_n1
            e_out = np.maximum(e_out, 0)
            e_out = np.dot(e_out, W_n2) + b_n2     # computing the vm
            # RL Expr
            # nexprs.append(E_Acts[np.argmax(e_out)])
            # SL Expr
            nexprs.append(e_out)

            if lh == 1:
                e_in_t = e_in
            else:   #lh==4
                e_in_t = np.concatenate([np.asarray(selected_e_ins[1-lh:]).flatten(), e_in])
                e_in_t = np.concatenate([np.asarray([0]*(np.maximum(0, lh*2*ncombs - len(e_in_t)))), e_in_t])

            e_in_t_list.append(e_in_t)

            score = s_predict(e_in_t, s_param)
            s_mask  = rg.binomial(n=1, p=s_eps, size=(1,))
            s_ra = rg.uniform(size=(n_as,), low=0.0, high=1.0)
            score = s_ra * (s_mask) + score * (1 - s_mask)   #80% possibility is random

            score_act.append(np.argmax(score))
            scores.append(S_Acts[np.argmax(score)])

        e_in_ts.append(np.asarray(e_in_t_list))
        scores = np.asarray(scores)

        max_idx = np.argmax(scores)
        # max_idx = np.where(scores == np.amax(scores))[0][-1]
        score_acts.append(score_act)
        selected_e_in_ts.append(e_in_t_list[max_idx])  #the input of max score

        expr = np.concatenate([expr[:max_idx], nexprs[max_idx:max_idx+1], expr[max_idx+2:]], axis=0)
        exprs_ts.append(nexprs)

        max_idxs[loop_idx] = max_idx
        loop_idx += 1

    fexpr = expr[0]  #the last result
    # r0 = np.argmax(fexpr) == target and fexpr[target] > 0.9
    # r0 = fexpr[target] + float(np.argmax(fexpr) == target )
    r0 = fexpr[target]  #target is 0 or 1  so r0 is the result in vm

    # r0 = 0.0
    # if fexpr[target] > 0.8:
    #     r0 = 1.0

    selected_e_in_ts = np.asarray(selected_e_in_ts)
    # if isTrain:
    #     er.remember(e_in_ts, score_acts, exprs_ts, max_idxs, r0, s_param, s_predict)

    # return max_idxs, fexpr, r0, score_acts
    return max_idxs, fexpr, r0, score_acts, e_in_ts, exprs_ts


if __name__ == '__main__':
    # fname = './logic_all_8.pkl'
    fname = './curriculum_sm.pkl'
    tx, ty, cs, mises = pickle.load(open(fname,'rb'))
    for i in range(len(mises)):
        print(str(i)+"--"+str(cs[i])+"--"+str(mises[i])+"--"+str(max(mises[i]))+'\n')
    print(tx[0], ty[0], cs[0])
    print(len(tx))
    # batch_size = 40*max(10, len(tx))
    batch_size = 400
    tbatch = min(1 + int(len(tx)/4), 16)
    start_epoch = 10

    # only train part
    part = 5000
    # part = 352
    
    s_eps = 0.8
    er = ExperienceReplay(max_memory=max(len(tx),100)*10000, discount=.9)
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

    with tf.Session() as sess:
        s_nn = Score_lh_NN(batch_size, init_v, lh=lh)

        s_logits = s_nn.next_score()
        s_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=s_nn.targets, predictions=s_logits))
        s_train_op = tf.train.AdamOptimizer(lr1).minimize(s_loss, var_list=s_nn.params)

        init = tf.global_variables_initializer()
        sess.run(init)

        # read good params
        params1 = pickle.load(open('./good_params_vm.pkl','rb'), encoding='iso-8859-1')
        W_n1, b_n1, W_n2, b_n2 = params1
        e_param = [W_n1, b_n1, W_n2, b_n2]

        s_param = sess.run((s_nn.params))
        with open('default_sm.pkl', 'wb') as f:
            pickle.dump((e_param, s_param), f)

        o_s_param = [np.copy(s_p) for s_p in s_param]

        best_tcorrect = 0
        for e in range(20000):

            correct = 0
            tcorrect = 0
            full = 0
            sum_r = 0.0
            avg_r = 0.0
            sum_loss = 0
            fails = []
            e_in_ts_list = []
            scores_list = []
            exprs_ts_list = []
            max_idxs_list = []
            for jdx in range(len(tx)):
                idx=jdx
                xs = train_x[idx]
                yt1 = train_y1[idx]
                yt2 = train_y2[idx]
                len_xs = len_xss[idx]

                ms, nexpr, r0, scores, e_in_ts, exprs_ts = compute_max_idxs(e_param, s_param, xs, len_xs, yt2, er, s_eps, True)
                #print('index', idx, 'reward', r0, cs[idx],  nexpr[:2], ms[:7])
                e_in_ts_list.append(e_in_ts)
                scores_list.append(scores)
                exprs_ts_list.append(exprs_ts)
                max_idxs_list.append(ms)
                sum_r += r0   #record all result
                avg_r += r0/len(tx)
                # train after 80 epochs and every 15 times or solve the last one
                if (jdx % tbatch == tbatch - 1 or jdx == len(tx) - 1) and e > start_epoch:

                    s_ins, s_targets = er.get_s_batch(s_param, o_s_param, batch_size=batch_size)
                    # print('s targets', s_ins[0], s_targets[0])
                    rs_logits, rs_loss, _ = sess.run((s_logits, s_loss, s_train_op), {s_nn.e_in:s_ins, s_nn.targets: s_targets})
                    s_param = sess.run((s_nn.params))  ##update and get the realtime weight
                    ###print('score logits', rs_logits[0], 'loss', rs_loss)
                    sum_loss += rs_loss
                    e1 = '1'
                    e2 = 'or'
                    # e3 = ')'
                    ein1 = np.zeros(ncombs)
                    ein2 = np.zeros(ncombs)
                    # ein3 = np.zeros(ncombs)
                    ein1[leaves.index(e1)]=1
                    ein2[leaves.index(e2)]=1
                    # ein3[leaves.index(e3)]=1
                    e_in_t = np.concatenate([ein1,ein2])
                    # ein4 = e_predict(e_in_t, e_param)
                    # e_in_t = np.concatenate([ein4,ein3])
                    score = s_predict(e_in_t, s_param)
                    expr = e_predict(e_in_t, e_param)  #vm model
                    ###print(e1, e2, leaves.index(e1), leaves.index(e2), score, np.amax(expr))
                    # ########
                    # for i in range(s_ins.shape[0]):
                    #     if np.sum(np.abs(s_ins[i] - e_in_t))  == 0:
                    #         print(s_targets[i])

            # test
            correct = 0
            tcorrect = 0
            full = 0
            for kdx in range(len(tx)):
                xs = train_x[kdx]
                yt1 = train_y1[kdx]
                yt2 = train_y2[kdx]
                len_xs = len_xss[kdx]

                ms, nexpr, r0, scores= compute_max_idxs(e_param, s_param, xs, len_xs, yt2, er, .0, False)[:4]

                #print('index', kdx, 'reward', r0, cs[kdx],  nexpr[:2], ms[:7])
                #print(scores[:7])
                if np.argmax(nexpr) == ty[kdx]:
                    correct += 1
                if nexpr[ty[kdx]] > 0.8:
                    tcorrect += 1
                full += 1

            # e1 = '0'
            # e2 = ')'
            # ein1 = np.zeros(ncombs)
            # ein2 = np.zeros(ncombs)
            # ein1[leaves.index(e1)]=1
            # ein2[leaves.index(e2)]=1
            # e_in_t = np.concatenate([ein1,ein2])
            # score = s_predict(e_in_t, s_param)
            # expr = e_predict(e_in_t, e_param)
            # print(e1, e2, leaves.index(e1), leaves.index(e2), score, np.amax(expr))

            if tcorrect == full:
                print('learn true correct')
                with open('params_sm.pkl', 'wb') as f:
                    pickle.dump((e_param, s_param), f)
                sys.exit()

            print('epoch', e, 'sum r', sum_r, 'avg r', avg_r, 'sum loss', sum_loss, 'correct', correct, 'true correct', tcorrect, 'full', full, 'len memory', er.s_memory.write)

            for pdx in range(len(exprs_ts_list)):
                e_in_ts = e_in_ts_list[pdx]  #state
                score_acts = scores_list[pdx]  #action
                exprs_ts = exprs_ts_list[pdx]
                max_idxs = max_idxs_list[pdx]
                er.remember(e_in_ts, score_acts, exprs_ts, max_idxs, avg_r, s_param, s_predict)

            logfile = open('log4', 'a')
            if best_tcorrect < tcorrect:
                logfile.write('update params!') 
                best_tcorrect = tcorrect
                with open('params_sm.pkl', 'wb') as f:
                    pickle.dump((e_param, s_param), f)
            logfile.write('epoch {0} correct {1}, true correct {2} full {3} sum r {4} sum loss {5} len memory {6} eps {7} \n'\
                    .format(e, correct, tcorrect, full, sum_r, sum_loss, er.s_memory.write, s_eps))
            logfile.close()

            if e % 1000 > start_epoch and s_eps > 0.01:
                s_eps = s_eps/1.1

            if e % 1000 == 0:
                s_eps = 0.8

            if e % 50 == 0:
                # update double dqn params
                s_param = sess.run((s_nn.params))
                o_s_param = [np.copy(s_p) for s_p in s_param]

            with open('params_sm.pkl', 'wb') as f:
                pickle.dump((e_param, s_param), f)



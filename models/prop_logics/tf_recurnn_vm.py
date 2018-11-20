#!/usr/bin/env python
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import random
import argparse
import numpy as np
import tensorflow as tf
import scipy
import scipy.misc
import pickle
import time
from tf_recurnn_nn import RecurNN

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

lr1 = 0.001
init_v = 100*lr1

def one_hot_preprocess(xs):
    new_xs = []
    for x in xs:
        new_xs.append(np.concatenate([one_hots[x], [0]*(ncombs-nsymbols)]))
    for i in range(maxlen - len(xs)):
        new_xs.append([0]*ncombs)
    return new_xs



def compute_exprs(params, expr, lens, mss):
    W_n1, b_n1, W_n2, b_n2 = params

    nexprs = []
    lens = lens[0]
    for max_idxs in mss:
        loop_idx = 0
        exprs = []
        texpr = np.copy(expr)
        while texpr.shape[0] > 1 + maxlen - lens:
            nexpr = []
            scores = []
            for i in range(0, lens - loop_idx - 1):
                e_in = np.concatenate([texpr[i], texpr[i+1]])

                c = np.copy(e_in)
                e_in[c < 0.1] = 0 
                e_in[c > 0.9] = 1 
            
                nnt0 = np.dot(e_in, W_n1) + b_n1
                nnt0 = np.maximum(nnt0, 0)
                nnt0 = np.dot(nnt0, W_n2) + b_n2

#                 c = np.copy(nnt0)
#                 nnt0[c < 0.1] = 0 
#                 nnt0[c > 0.9] = 1 
            
                nexpr.append(nnt0)

            nexpr = np.asarray(nexpr)

            max_idx = max_idxs[loop_idx]
            loop_idx += 1

            nexpr = nexpr[max_idx:max_idx+1]

            texpr = np.concatenate([texpr[:max_idx], nexpr, texpr[max_idx+2:]], axis=0)

        nexprs.append(texpr[0])

    return nexprs


def all_combs(mss, idx, lens):
    lensarr = np.arange(lens)
    fact = int(scipy.misc.factorial(lens-1))
    jdx = 0
    for i in range(mss.shape[0]):
        mss[i][idx] = lensarr[jdx]
        if i % fact == fact - 1:
            jdx += 1
        if jdx == len(lensarr):
            jdx = 0
    return mss

if __name__ == '__main__':
    # fname = './logic_all_8.pkl'
    fname = './curriculum_vm.pkl'
    tx, ty, cs = pickle.load(open(fname))[:3]
    print(tx, ty[0], cs[0])

    batch = 8

    # only train part
    part = 5000
    tx = tx[:part]
    ty = ty[:part]
    cs = cs[:part]
    train_set = [tx, ty]

    train_x = []
    train_y1 = []
    train_y2 = []
    len_xss = []
    mss_s = []

    n_combs = 0
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
        
        lenf = len(x) - 1
        fact = int(scipy.misc.factorial(lenf))
        mss = np.zeros((fact, lenf), dtype=np.int32)
        for idx in range(mss.shape[1]):
            mss = all_combs(mss, idx, lenf - idx)

        # mss = np.zeros((1, lenf), dtype=np.int32)
        mss_s.append(mss)
        n_combs += fact

    print(len(tx), n_combs)
    # sys.exit()

    train_x = np.asarray(train_x)
    train_y1 = np.asarray(train_y1)
    train_y2 = np.asarray(train_y2)
    len_xss = np.asarray(len_xss)
    # mss_s = np.asarray(mss_s)

    with tf.Session() as sess:
        classifier = RecurNN(init_v, v_epsilon)

        # init = tf.global_variables_initializer()
        # sess.run(init)

        logits, max_idxs, loop_idx, len_expr, s0 = classifier.forward_pass()        
        
        smed = tf.nn.softmax(logits)

        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=classifier.target1, logits=logits))
        # loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=classifier.target2, logits=logits))
        loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=classifier.target1, predictions=logits))

        optimizer = tf.train.AdamOptimizer(learning_rate=lr1)
        gvs = optimizer.compute_gradients(loss=loss, var_list=[classifier.W_n1, classifier.b_n1, classifier.W_n2, classifier.b_n2])
        # capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        capped_gvs = gvs
        gwn1 = tf.zeros((2*ncombs,ncombs*ncombs))
        gbn1 = tf.zeros((ncombs*ncombs))
        gwn2 = tf.zeros((ncombs*ncombs,ncombs))
        gbn2 = tf.zeros((ncombs))
        gws1 = tf.zeros((2*ncombs,ncombs*ncombs))
        gbs1 = tf.zeros((ncombs*ncombs))
        gws2 = tf.zeros((ncombs*ncombs,1))
        gbs2 = tf.zeros((1))

        batched_gv = [(gwn1, gvs[0][1]), (gbn1, gvs[1][1]), (gwn2, gvs[2][1]), (gbn2, gvs[3][1])]

        train_op = optimizer.apply_gradients(batched_gv)

        init = tf.global_variables_initializer()
        sess.run(init)

        # read good params
        params1 = pickle.load(open('./good_params_vm.pkl'))
        W_n1, b_n1, W_n2, b_n2= params1
        sess.run(classifier.W_n1.assign(W_n1))
        sess.run(classifier.b_n1.assign(b_n1))
        sess.run(classifier.W_n2.assign(W_n2))
        sess.run(classifier.b_n2.assign(b_n2))

        params = sess.run((classifier.params))
        with open('default_vm.pkl', 'wb') as f:
            pickle.dump(params, f)

        correct_count = 0
        best_sum_loss = np.inf
        for e in range(2000):
            print('epoch', e)

            correct = 0
            full = 0
            for idx in range(len(tx)):
                xs = train_x[idx]
                yt1 = train_y1[idx]
                yt2 = train_y2[idx]
                len_xs = len_xss[idx]
                mss = mss_s[idx]

                params = sess.run((classifier.params))
                nexprs = compute_exprs(params, xs, len_xs, mss)

                for jdx,nexpr in enumerate(nexprs):
                    print('index', idx, cs[idx], nexpr[:2], mss[jdx][:7])
                    # print('logits', nexpr)
                    if np.argmax(nexpr) == ty[idx]:
                        correct += 1
                full += len(nexprs)
            print('correct', correct, 'full', full)

            avg_g = np.asarray([np.zeros_like(pa) for pa in params])

            sum_loss = 0
            learned_ac = 0
            sum_g = 0
            jdx = 0
            for idx in range(len(tx)):
                xs = train_x[idx]
                yt1 = train_y1[idx]
                yt2 = train_y2[idx]
                len_xs = len_xss[idx]
                mss = mss_s[idx]
                
                for ms in mss:
                    ms = np.concatenate([ms, [0]*(maxlen -len_xs[0])])
                    params = sess.run((classifier.params))

                    r_logits, r_loss, r_grad, r_max_idxs, r_loop_idx, r_len_expr, read_in, r_s0, r_smed = \
                    sess.run((logits, loss, capped_gvs, max_idxs, loop_idx, len_expr, classifier.expr, smed, s0), \
                    {classifier.expr:xs, classifier.target1: yt1, classifier.target2: yt2, classifier.lens:len_xs, classifier.max_idxs:ms})

                    r_gs = np.asarray([g for g, v in r_grad])
                    avg_g = r_gs/float(batch) + avg_g

                    sum_loss += r_loss
                    if np.argmax(r_logits[0]) == ty[idx]:
                        learned_ac += 1
                    print('index',  idx, cs[idx], yt1, yt2)
                    print('logits', r_logits[0], r_smed, r_loss, r_s0)
                    print('loop index', r_loop_idx, 'len expr', r_len_expr, 'r max idxs', ms)
                    print('grad', [np.sum(np.abs(rg)) for rg,v in r_grad])

                    if jdx % batch == batch - 1 or jdx == len(tx) - 1:

                        print('avg grad', [np.sum(np.abs(rg)) for rg in avg_g])
                        _ = sess.run((train_op), \
                        {gwn1:avg_g[0], gbn1:avg_g[1], gwn2:avg_g[2], gbn2:avg_g[3]})

                        sum_g += np.sum(np.asarray([np.sum(np.abs(rg)) for rg in avg_g]))
                        avg_g = [np.zeros_like(pa) for pa in params] 
                    jdx += 1

            logfile = open('log1', 'a')
            logfile.write('epoch {0} correct {1} full {2} loss {3} learned correct {4} sum_g {5} \n'\
                    .format(e, correct, full, sum_loss, learned_ac, sum_g))

            if correct == full and sum_loss < 0.006:
                print('learn correct')
                break

            # save at every epoch
            if sum_loss < best_sum_loss:
                logfile.write('save params {0} {1} !\n'.format(sum_loss, best_sum_loss))
                best_sum_loss = sum_loss
                params = sess.run((classifier.params))
                with open('params_vm.pkl', 'wb') as f:
                    pickle.dump(params, f)
            logfile.close()



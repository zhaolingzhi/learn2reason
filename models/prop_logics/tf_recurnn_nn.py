#!/usr/bin/env python
import sys
import os
import numpy as np
import tensorflow as tf

maxlen = 30
nsymbols = 7
ncombs = 37

class RecurNN(object):

    def __init__(self, init_v, v_epsilon):
        self.W_n1 = tf.get_variable("W_n1",[2*ncombs,ncombs*ncombs],initializer=tf.random_uniform_initializer(-init_v,init_v))
        # self.W_n1 = tf.get_variable("W_n1",[2*ncombs,ncombs*ncombs],initializer=tf.contrib.layers.xavier_initializer())
        self.b_n1 = tf.get_variable("b_n1",[ncombs*ncombs,],initializer=tf.random_uniform_initializer(0,0))
        self.W_n2 = tf.get_variable("W_n2",[ncombs*ncombs,ncombs],initializer=tf.random_uniform_initializer(-init_v,init_v))
        self.b_n2 = tf.get_variable("b_n2",[ncombs],initializer=tf.random_uniform_initializer(0,0))

        self.W_s1 = tf.get_variable("W_s1",[2*ncombs,ncombs*ncombs],initializer=tf.random_uniform_initializer(-init_v,init_v))
        self.b_s1 = tf.get_variable("b_s1",[ncombs*ncombs,],initializer=tf.random_uniform_initializer(0,0))
        self.W_s2 = tf.get_variable("W_s2",[ncombs*ncombs, 1],initializer=tf.random_uniform_initializer(-init_v,init_v))
        self.b_s2 = tf.get_variable("b_s2",[1],initializer=tf.random_uniform_initializer(0,0))
        self.expr = tf.placeholder(tf.float32,  shape=(maxlen, ncombs))
        self.max_idxs = tf.placeholder(tf.int32,  shape=(maxlen-1,))
        self.target1 = tf.placeholder(tf.int32,  shape=(1, ncombs))
        self.target2 = tf.placeholder(tf.int32,  shape=(1, ))
        self.lens = tf.placeholder(tf.int32,  shape=(1, ))
        self.symbol0 =tf.Variable(np.zeros((1, ncombs), dtype=np.float32), name='symbol0') 
        self.v_epsilon = v_epsilon
        
        self.params = [self.W_n1, self.b_n1, self.W_n2, self.b_n2]

    def forward_pass(self):
        max_idxs = self.max_idxs
        expr = self.expr
        # len_expr = tf.shape(max_idxs).shape[0]
        len_expr = self.lens[0]
        loop_idx = 0
        s0 = tf.ones([1,1])

        def _one_pass(expr, max_idxs, loop_idx, len_expr, s0):

            max_idx = max_idxs[loop_idx]

            def _calc(e_in):
                e_in = tf.reshape(e_in, [-1])
                e_in = tf.expand_dims(e_in, 0)
                ne = tf.matmul(e_in, self.W_n1) + self.b_n1
                # ne = tf.nn.leaky_relu(ne, 0.01)
                ne = tf.nn.relu(ne)
                ne = tf.matmul(ne, self.W_n2) + self.b_n2
                # ne = tf.tanh(ne)
                # ne = tf.nn.softmax(ne)

                return ne

            elems = (expr[:-1], expr[1:])
            nexprs = tf.map_fn(_calc, elems, dtype=tf.float32)

            nexpr = nexprs[max_idx]

            # nexpr = tf.nn.softmax(nexpr)
            
            # m, v = tf.nn.moments(nexpr, axes=[1])
            # nexpr = tf.nn.batch_normalization(nexpr, mean=m, variance=v, offset=None, scale=None, variance_epsilon=self.v_epsilon)

            expr = tf.concat([expr[:max_idx], nexpr, expr[max_idx+2:], self.symbol0], axis=0)

            expr = tf.reshape(expr, [maxlen, ncombs])
            len_expr = tf.subtract(len_expr, 1)
            loop_idx = tf.add(loop_idx,1)
            return expr, max_idxs, loop_idx, len_expr, s0

        loop_cond = lambda expr, max_idxs, loop_idx, len_expr, s0: tf.less(1, len_expr) 
        loop_vars = [expr, max_idxs, loop_idx, len_expr, s0]
        expr, max_idxs, loop_idx, len_expr, s0 =tf.while_loop(loop_cond, _one_pass, loop_vars)

        expr = expr[:1]

        # m, v = tf.nn.moments(expr, axes=[1])
        # expr = tf.nn.batch_normalization(expr, mean=m, variance=v,\
        #         offset=None, scale=None, variance_epsilon=1e-6)

        return expr, max_idxs, loop_idx, len_expr, s0

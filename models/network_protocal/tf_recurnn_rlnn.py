#!/usr/bin/env python
import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten


maxlen = 30
nsymbols = 17
ncombs = 40
na_e = 12
nactions = 4
slim = tf.contrib.slim


class Score_lh_NN2(object):

    def __init__(self, batch_size, init_v, v_epsilon=1e-9, lh=4, n_as=5):
        self.W_s1 = tf.get_variable("W_s1",[lh*2*ncombs,lh*2*ncombs],initializer=tf.random_uniform_initializer(-init_v,init_v))
        self.b_s1 = tf.get_variable("b_s1",[lh*2*ncombs,],initializer=tf.random_uniform_initializer(0,0))
        self.W_s2 = tf.get_variable("W_s2",[lh*2*ncombs,n_as],initializer=tf.random_uniform_initializer(-init_v,init_v))
        self.b_s2 = tf.get_variable("b_s2",[n_as],initializer=tf.random_uniform_initializer(0,0))

        self.e_in = tf.placeholder(tf.float32,  shape=(batch_size, lh*2*ncombs))
        self.targets = tf.placeholder(tf.float32,  shape=(batch_size, n_as))
        
        self.params = [self.W_s1, self.b_s1, self.W_s2, self.b_s2]

    def next_score(self):
        score = tf.matmul(self.e_in, self.W_s1) + self.b_s1
        score = tf.nn.relu(score)
        score = tf.matmul(score, self.W_s2) + self.b_s2
        return score

class Score_light_lh_NN(object):

    def __init__(self, batch_size, init_v, v_epsilon=1e-9, lh=4, n_as=5):
        self.W_s2 = tf.get_variable("W_s2",[lh*ncombs*ncombs, lh*2*ncombs],initializer=tf.random_uniform_initializer(-init_v,init_v))
        self.b_s2 = tf.get_variable("b_s2",[lh*2*ncombs,],initializer=tf.random_uniform_initializer(0,0))
        self.W_s3 = tf.get_variable("W_s3",[lh*2*ncombs,2*n_as],initializer=tf.random_uniform_initializer(-init_v,init_v))
        self.b_s3 = tf.get_variable("b_s3",[2*n_as],initializer=tf.random_uniform_initializer(0,0))
        self.W_s4 = tf.get_variable("W_s4",[2*n_as,n_as],initializer=tf.random_uniform_initializer(-init_v,init_v))
        self.b_s4 = tf.get_variable("b_s4",[n_as],initializer=tf.random_uniform_initializer(0,0))

        self.e_in = tf.placeholder(tf.float32,  shape=(batch_size, lh*ncombs*ncombs))
        self.targets = tf.placeholder(tf.float32,  shape=(batch_size, n_as))
        
        self.params = [self.W_s2, self.b_s2, self.W_s3, self.b_s3, self.W_s4, self.b_s4]

    def next_score(self):
        score = tf.matmul(self.e_in, self.W_s2) + self.b_s2
        score = tf.nn.relu(score)
        score = tf.matmul(score, self.W_s3) + self.b_s3
        score = tf.nn.relu(score)
        score = tf.matmul(score, self.W_s4) + self.b_s4
        return score

class ExprNN(object):

    def __init__(self, batch_size, init_v, v_epsilon=1e-9):
        self.W_n1 = tf.get_variable("W_n1",[2*ncombs,ncombs*ncombs],initializer=tf.random_uniform_initializer(-init_v,init_v))
        self.b_n1 = tf.get_variable("b_n1",[ncombs*ncombs,],initializer=tf.random_uniform_initializer(0,0))
        self.W_n2 = tf.get_variable("W_n2",[ncombs*ncombs,na_e],initializer=tf.random_uniform_initializer(-init_v,init_v))
        self.b_n2 = tf.get_variable("b_n2",[na_e],initializer=tf.random_uniform_initializer(0,0))

        self.e_in = tf.placeholder(tf.float32,  shape=(batch_size, 2*ncombs))
        self.targets = tf.placeholder(tf.float32,  shape=(batch_size, na_e))
        
        self.params = [self.W_n1, self.b_n1, self.W_n2, self.b_n2]

    def next_expr(self):
        # e_in = tf.expand_dims(e_in, 0)
        ne = tf.matmul(self.e_in, self.W_n1) + self.b_n1
        ne = tf.nn.relu(ne)
        ne = tf.matmul(ne, self.W_n2) + self.b_n2
        return ne


class ScoreNN(object):

    def __init__(self, batch_size, init_v, v_epsilon=1e-9, n_as=5):
        self.W_s1 = tf.get_variable("W_s1",[2*ncombs,ncombs*ncombs],initializer=tf.random_uniform_initializer(-init_v,init_v))
        self.b_s1 = tf.get_variable("b_s1",[ncombs*ncombs,],initializer=tf.random_uniform_initializer(0,0))
        self.W_s2 = tf.get_variable("W_s2",[ncombs*ncombs,n_as],initializer=tf.random_uniform_initializer(-init_v,init_v))
        self.b_s2 = tf.get_variable("b_s2",[n_as],initializer=tf.random_uniform_initializer(0,0))

        self.e_in = tf.placeholder(tf.float32,  shape=(batch_size, 2*ncombs))
        self.targets = tf.placeholder(tf.float32,  shape=(batch_size, n_as))
        
        self.params = [self.W_s1, self.b_s1, self.W_s2, self.b_s2]

    def next_score(self):
        score = tf.matmul(self.e_in, self.W_s1) + self.b_s1
        score = tf.nn.relu(score)
        score = tf.matmul(score, self.W_s2) + self.b_s2
        return score

class Expr_lh_NN(object):

    def __init__(self, batch_size, init_v, v_epsilon=1e-9):
        self.W_n1 = tf.get_variable("W_n1",[8*ncombs,4*ncombs*ncombs],initializer=tf.random_uniform_initializer(-init_v,init_v))
        self.b_n1 = tf.get_variable("b_n1",[4*ncombs*ncombs,],initializer=tf.random_uniform_initializer(0,0))
        self.W_n2 = tf.get_variable("W_n2",[4*ncombs*ncombs,na_e],initializer=tf.random_uniform_initializer(-init_v,init_v))
        self.b_n2 = tf.get_variable("b_n2",[na_e],initializer=tf.random_uniform_initializer(0,0))

        self.e_in = tf.placeholder(tf.float32,  shape=(batch_size, 8*ncombs))
        self.targets = tf.placeholder(tf.float32,  shape=(batch_size, na_e))
        
        self.params = [self.W_n1, self.b_n1, self.W_n2, self.b_n2]

    def next_expr(self):
        # e_in = tf.expand_dims(e_in, 0)
        ne = tf.matmul(self.e_in, self.W_n1) + self.b_n1
        ne = tf.nn.relu(ne)
        ne = tf.matmul(ne, self.W_n2) + self.b_n2
        return ne


class Score_lh_NN(object):

    def __init__(self, batch_size, init_v, v_epsilon=1e-9, lh=4, n_as=5):
        self.W_s1 = tf.get_variable("W_s1",[lh*2*ncombs,lh*ncombs*ncombs],initializer=tf.random_uniform_initializer(-init_v,init_v))
        self.b_s1 = tf.get_variable("b_s1",[lh*ncombs*ncombs,],initializer=tf.random_uniform_initializer(0,0))
        self.W_s2 = tf.get_variable("W_s2",[lh*ncombs*ncombs,n_as],initializer=tf.random_uniform_initializer(-init_v,init_v))
        self.b_s2 = tf.get_variable("b_s2",[n_as],initializer=tf.random_uniform_initializer(0,0))

        self.e_in = tf.placeholder(tf.float32,  shape=(batch_size, lh*2*ncombs))
        self.targets = tf.placeholder(tf.float32,  shape=(batch_size, n_as))
        
        self.params = [self.W_s1, self.b_s1, self.W_s2, self.b_s2]

    def next_score(self):
        score = tf.matmul(self.e_in, self.W_s1) + self.b_s1
        score = tf.nn.relu(score)
        score = tf.matmul(score, self.W_s2) + self.b_s2
        return score

class Score_light_lh_NN(object):

    def __init__(self, batch_size, init_v, v_epsilon=1e-9, lh=4, n_as=5):
        self.W_s2 = tf.get_variable("W_s2",[lh*ncombs*ncombs, lh*2*ncombs],initializer=tf.random_uniform_initializer(-init_v,init_v))
        self.b_s2 = tf.get_variable("b_s2",[lh*2*ncombs,],initializer=tf.random_uniform_initializer(0,0))
        self.W_s3 = tf.get_variable("W_s3",[lh*2*ncombs,2*n_as],initializer=tf.random_uniform_initializer(-init_v,init_v))
        self.b_s3 = tf.get_variable("b_s3",[2*n_as],initializer=tf.random_uniform_initializer(0,0))
        self.W_s4 = tf.get_variable("W_s4",[2*n_as,n_as],initializer=tf.random_uniform_initializer(-init_v,init_v))
        self.b_s4 = tf.get_variable("b_s4",[n_as],initializer=tf.random_uniform_initializer(0,0))

        self.e_in = tf.placeholder(tf.float32,  shape=(batch_size, lh*ncombs*ncombs))
        self.targets = tf.placeholder(tf.float32,  shape=(batch_size, n_as))
        
        self.params = [self.W_s2, self.b_s2, self.W_s3, self.b_s3, self.W_s4, self.b_s4]

    def next_score(self):
        score = tf.matmul(self.e_in, self.W_s2) + self.b_s2
        score = tf.nn.relu(score)
        score = tf.matmul(score, self.W_s3) + self.b_s3
        score = tf.nn.relu(score)
        score = tf.matmul(score, self.W_s4) + self.b_s4
        return score


class Nscore_lh_NN(object):

    def __init__(self, batch_size, init_v, v_epsilon=1e-9, lh=4):
        self.W_a1 = tf.get_variable("W_s1",[nsymbols*4, nsymbols**4],initializer=tf.random_uniform_initializer(-init_v,init_v))
        self.b_a1 = tf.get_variable("b_s1",[nsymbols**4,],initializer=tf.random_uniform_initializer(0,0))
        self.W_a2 = tf.get_variable("W_s2",[nsymbols**4,nactions],initializer=tf.random_uniform_initializer(-init_v,init_v))
        self.b_a2 = tf.get_variable("b_s2",[nactions],initializer=tf.random_uniform_initializer(0,0))

        self.e_in = tf.placeholder(tf.float32,  shape=(batch_size, nsymbols*4))
        self.targets = tf.placeholder(tf.float32,  shape=(batch_size, nactions))
        
        self.params = [self.W_a1, self.b_a1, self.W_a2, self.b_a2]

    def next_score(self):
        score = tf.matmul(self.e_in, self.W_a1) + self.b_a1
        score = tf.nn.relu(score)
        score = tf.matmul(score, self.W_a2) + self.b_a2
        return score


class Nscore_lh_leNN(object):

    def __init__(self, batch_size, init_v, v_epsilon=1e-9, lh=4):
 	self.conv1_w = tf.Variable(tf.truncated_normal(shape = [5,5,1,6],mean = 0, stddev = init_v))
 	self.conv1_b = tf.Variable(tf.zeros(6))
 	self.conv2_w = tf.Variable(tf.truncated_normal(shape = [5,5,6,16], mean = 0, stddev = init_v))
 	self.conv2_b = tf.Variable(tf.zeros(16))
 	self.fc1_w = tf.Variable(tf.truncated_normal(shape = (2704,120), mean = 0, stddev = init_v))
 	self.fc1_b = tf.Variable(tf.zeros(120))
 	self.fc2_w = tf.Variable(tf.truncated_normal(shape = (120,84), mean = 0, stddev = init_v))
 	self.fc2_b = tf.Variable(tf.zeros(84))
 	self.fc3_w = tf.Variable(tf.truncated_normal(shape = (84,nactions), mean = 0 , stddev = init_v))
 	self.fc3_b = tf.Variable(tf.zeros(nactions))

        self.e_in = tf.placeholder(tf.float32,  shape=(None, nsymbols*nsymbols, nsymbols*nsymbols, 1))
        self.targets = tf.placeholder(tf.float32,  shape=(batch_size, nactions))
        
        self.params = [self.conv1_w, self.conv1_b, self.conv2_w, self.conv2_b, \
		self.fc1_w, self.fc1_b, self.fc2_w, self.fc2_b, self.fc3_w, self.fc3_b]

    def next_score(self):
 	conv1 = tf.nn.conv2d(self.e_in, self.conv1_w, strides = [1,1,1,1], padding = 'VALID') + self.conv1_b 
 	conv1 = tf.nn.relu(conv1)
 	pool_1 = tf.nn.max_pool(conv1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
 	conv2 = tf.nn.conv2d(pool_1, self.conv2_w, strides = [1,1,1,1], padding = 'VALID') + self.conv2_b
 	conv2 = tf.nn.relu(conv2)
 	pool_2 = tf.nn.max_pool(conv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
 	fc1 = flatten(pool_2)
 	fc1 = tf.matmul(fc1,self.fc1_w) + self.fc1_b
 	fc1 = tf.nn.relu(fc1)
 	fc2 = tf.matmul(fc1,self.fc2_w) + self.fc2_b
 	fc2 = tf.nn.relu(fc2)
 	logits = tf.matmul(fc2, self.fc3_w) + self.fc3_b
 	return logits

    def set_params(self, sess, params):
        conv1_w, conv1_b, conv2_w, conv2_b, fc1_w, fc1_b, fc2_w, fc2_b, fc3_w, fc3_b = params
        sess.run(self.conv1_w.assign(conv1_w))
        sess.run(self.conv1_b.assign(conv1_b))
        sess.run(self.conv2_w.assign(conv2_w))
        sess.run(self.conv2_b.assign(conv2_b))
        sess.run(self.fc1_w.assign(fc1_w))
        sess.run(self.fc1_b.assign(fc1_b))
        sess.run(self.fc2_w.assign(fc2_w))
        sess.run(self.fc2_b.assign(fc2_b))
        sess.run(self.fc3_w.assign(fc3_w))
        sess.run(self.fc3_b.assign(fc3_b))


class Score_lh_leNN(object):

    def __init__(self, batch_size, init_v, v_epsilon=1e-9, lh=4):
 	self.conv1_w = tf.Variable(tf.truncated_normal(shape = [5,5,1,6],mean = 0, stddev = init_v))
 	self.conv1_b = tf.Variable(tf.zeros(6))
 	self.conv2_w = tf.Variable(tf.truncated_normal(shape = [5,5,6,16], mean = 0, stddev = init_v))
 	self.conv2_b = tf.Variable(tf.zeros(16))
 	self.fc1_w = tf.Variable(tf.truncated_normal(shape = (576,120), mean = 0, stddev = init_v))
 	self.fc1_b = tf.Variable(tf.zeros(120))
 	self.fc2_w = tf.Variable(tf.truncated_normal(shape = (120,84), mean = 0, stddev = init_v))
 	self.fc2_b = tf.Variable(tf.zeros(84))
 	self.fc3_w = tf.Variable(tf.truncated_normal(shape = (84,n_as), mean = 0 , stddev = init_v))
 	self.fc3_b = tf.Variable(tf.zeros(n_as))

        self.e_in = tf.placeholder(tf.float32,  shape=(None, ncombs, ncombs, 1))
        self.targets = tf.placeholder(tf.float32,  shape=(batch_size, n_as))
        
        self.params = [self.conv1_w, self.conv1_b, self.conv2_w, self.conv2_b, \
		self.fc1_w, self.fc1_b, self.fc2_w, self.fc2_b, self.fc3_w, self.fc3_b]

    def next_score(self):
 	conv1 = tf.nn.conv2d(self.e_in, self.conv1_w, strides = [1,1,1,1], padding = 'VALID') + self.conv1_b 
 	conv1 = tf.nn.relu(conv1)
 	pool_1 = tf.nn.max_pool(conv1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
 	conv2 = tf.nn.conv2d(pool_1, self.conv2_w, strides = [1,1,1,1], padding = 'VALID') + self.conv2_b
 	conv2 = tf.nn.relu(conv2)
 	pool_2 = tf.nn.max_pool(conv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
 	fc1 = flatten(pool_2)
 	fc1 = tf.matmul(fc1,self.fc1_w) + self.fc1_b
 	fc1 = tf.nn.relu(fc1)
 	fc2 = tf.matmul(fc1,self.fc2_w) + self.fc2_b
 	fc2 = tf.nn.relu(fc2)
 	logits = tf.matmul(fc2, self.fc3_w) + self.fc3_b
 	return logits

    def set_params(self, sess, params):
        conv1_w, conv1_b, conv2_w, conv2_b, fc1_w, fc1_b, fc2_w, fc2_b, fc3_w, fc3_b = params
        sess.run(self.conv1_w.assign(conv1_w))
        sess.run(self.conv1_b.assign(conv1_b))
        sess.run(self.conv2_w.assign(conv2_w))
        sess.run(self.conv2_b.assign(conv2_b))
        sess.run(self.fc1_w.assign(fc1_w))
        sess.run(self.fc1_b.assign(fc1_b))
        sess.run(self.fc2_w.assign(fc2_w))
        sess.run(self.fc2_b.assign(fc2_b))
        sess.run(self.fc3_w.assign(fc3_w))
        sess.run(self.fc3_b.assign(fc3_b))



class Nscore_lh_alexNN(object):

    def __init__(self, batch_size, init_v, v_epsilon=1e-9, lh=4):
 	self.conv1_w = tf.Variable(tf.truncated_normal(shape = [11,1,1,64],mean = 0, stddev = init_v))
 	self.conv1_b = tf.Variable(tf.zeros(64))
 	self.conv2_w = tf.Variable(tf.truncated_normal(shape = [5,5,64,192], mean = 0, stddev = init_v))
 	self.conv2_b = tf.Variable(tf.zeros(192))
 	self.conv3_w = tf.Variable(tf.truncated_normal(shape = [3,3,192,384], mean = 0, stddev = init_v))
 	self.conv3_b = tf.Variable(tf.zeros(384))
 	self.conv4_w = tf.Variable(tf.truncated_normal(shape = [3,3,384,384], mean = 0, stddev = init_v))
 	self.conv4_b = tf.Variable(tf.zeros(384))
 	self.conv4_w = tf.Variable(tf.truncated_normal(shape = [3,3,384,256], mean = 0, stddev = init_v))
 	self.conv4_b = tf.Variable(tf.zeros(256))
 	self.fc1_w = tf.Variable(tf.truncated_normal(shape = (2704,120), mean = 0, stddev = init_v))
 	self.fc1_b = tf.Variable(tf.zeros(120))
 	self.fc2_w = tf.Variable(tf.truncated_normal(shape = (120,84), mean = 0, stddev = init_v))
 	self.fc2_b = tf.Variable(tf.zeros(84))
 	self.fc3_w = tf.Variable(tf.truncated_normal(shape = (84,nactions), mean = 0 , stddev = init_v))
 	self.fc3_b = tf.Variable(tf.zeros(nactions))

        self.e_in = tf.placeholder(tf.float32,  shape=(None, nsymbols*nsymbols, nsymbols*nsymbols, 1))
        self.targets = tf.placeholder(tf.float32,  shape=(batch_size, nactions))
        
        self.params = [self.conv1_w, self.conv1_b, self.conv2_w, self.conv2_b, \
		self.fc1_w, self.fc1_b, self.fc2_w, self.fc2_b, self.fc3_w, self.fc3_b]

    def next_score(self):
 	conv1 = tf.nn.conv2d(self.e_in, self.conv1_w, strides = [1,1,1,1], padding = 'VALID') + self.conv1_b 
 	conv1 = tf.nn.relu(conv1)
 	pool_1 = tf.nn.max_pool(conv1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
 	conv2 = tf.nn.conv2d(pool_1, self.conv2_w, strides = [1,1,1,1], padding = 'VALID') + self.conv2_b
 	conv2 = tf.nn.relu(conv2)
 	pool_2 = tf.nn.max_pool(conv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
 	fc1 = flatten(pool_2)
 	fc1 = tf.matmul(fc1,self.fc1_w) + self.fc1_b
 	fc1 = tf.nn.relu(fc1)
 	fc2 = tf.matmul(fc1,self.fc2_w) + self.fc2_b
 	fc2 = tf.nn.relu(fc2)
 	logits = tf.matmul(fc2, self.fc3_w) + self.fc3_b
 	return logits

    def set_params(self, sess, params):
        conv1_w, conv1_b, conv2_w, conv2_b, fc1_w, fc1_b, fc2_w, fc2_b, fc3_w, fc3_b = params
        sess.run(self.conv1_w.assign(conv1_w))
        sess.run(self.conv1_b.assign(conv1_b))
        sess.run(self.conv2_w.assign(conv2_w))
        sess.run(self.conv2_b.assign(conv2_b))
        sess.run(self.fc1_w.assign(fc1_w))
        sess.run(self.fc1_b.assign(fc1_b))
        sess.run(self.fc2_w.assign(fc2_w))
        sess.run(self.fc2_b.assign(fc2_b))
        sess.run(self.fc3_w.assign(fc3_w))
        sess.run(self.fc3_b.assign(fc3_b))
class Nscore_lh_alexNN(object):

    def __init__(self, batch_size, init_v, v_epsilon=1e-9, lh=4):
 	self.conv1_w = tf.Variable(tf.truncated_normal(shape = [11,1,1,64],mean = 0, stddev = init_v))
 	self.conv1_b = tf.Variable(tf.zeros(64))
 	self.conv2_w = tf.Variable(tf.truncated_normal(shape = [5,5,64,192], mean = 0, stddev = init_v))
 	self.conv2_b = tf.Variable(tf.zeros(192))
 	self.conv3_w = tf.Variable(tf.truncated_normal(shape = [3,3,192,384], mean = 0, stddev = init_v))
 	self.conv3_b = tf.Variable(tf.zeros(384))
 	self.conv4_w = tf.Variable(tf.truncated_normal(shape = [3,3,384,384], mean = 0, stddev = init_v))
 	self.conv4_b = tf.Variable(tf.zeros(384))
 	self.conv4_w = tf.Variable(tf.truncated_normal(shape = [3,3,384,256], mean = 0, stddev = init_v))
 	self.conv4_b = tf.Variable(tf.zeros(256))
 	self.fc1_w = tf.Variable(tf.truncated_normal(shape = (2704,120), mean = 0, stddev = init_v))
 	self.fc1_b = tf.Variable(tf.zeros(120))
 	self.fc2_w = tf.Variable(tf.truncated_normal(shape = (120,84), mean = 0, stddev = init_v))
 	self.fc2_b = tf.Variable(tf.zeros(84))
 	self.fc3_w = tf.Variable(tf.truncated_normal(shape = (84,nactions), mean = 0 , stddev = init_v))
 	self.fc3_b = tf.Variable(tf.zeros(nactions))

        self.e_in = tf.placeholder(tf.float32,  shape=(None, nsymbols*nsymbols, nsymbols*nsymbols, 1))
        self.targets = tf.placeholder(tf.float32,  shape=(batch_size, nactions))
        
        self.params = [self.conv1_w, self.conv1_b, self.conv2_w, self.conv2_b, \
		self.fc1_w, self.fc1_b, self.fc2_w, self.fc2_b, self.fc3_w, self.fc3_b]

    def next_score(self):
 	conv1 = tf.nn.conv2d(self.e_in, self.conv1_w, strides = [1,1,1,1], padding = 'VALID') + self.conv1_b 
 	conv1 = tf.nn.relu(conv1)
 	pool_1 = tf.nn.max_pool(conv1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
 	conv2 = tf.nn.conv2d(pool_1, self.conv2_w, strides = [1,1,1,1], padding = 'VALID') + self.conv2_b
 	conv2 = tf.nn.relu(conv2)
 	pool_2 = tf.nn.max_pool(conv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
 	fc1 = flatten(pool_2)
 	fc1 = tf.matmul(fc1,self.fc1_w) + self.fc1_b
 	fc1 = tf.nn.relu(fc1)
 	fc2 = tf.matmul(fc1,self.fc2_w) + self.fc2_b
 	fc2 = tf.nn.relu(fc2)
 	logits = tf.matmul(fc2, self.fc3_w) + self.fc3_b
 	return logits

    def set_params(self, sess, params):
        conv1_w, conv1_b, conv2_w, conv2_b, fc1_w, fc1_b, fc2_w, fc2_b, fc3_w, fc3_b = params
        sess.run(self.conv1_w.assign(conv1_w))
        sess.run(self.conv1_b.assign(conv1_b))
        sess.run(self.conv2_w.assign(conv2_w))
        sess.run(self.conv2_b.assign(conv2_b))
        sess.run(self.fc1_w.assign(fc1_w))
        sess.run(self.fc1_b.assign(fc1_b))

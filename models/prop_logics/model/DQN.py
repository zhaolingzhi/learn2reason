#! /usr/bin/env python
# coding=utf-8

import random
from collections import deque

import keras
import pickle
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.initializers import random_uniform
from models.prop_logics.SumTree import SumTree

np.random.seed(1)
weight_file = "weight.h5"

prq_e= 1e-6


class DQN:
    def __init__(
            self,
            state_size,
            action_size,
            epsilon=1,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            learning_rate=0.001,
            gamma=0.9,
            test=False):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.memory = SumTree(100000)
        self.test = test
        self.count = 0
        self.max_count = 50
        self.model = self._build_model()
        self.model_target = self._build_model()
        self.model_target.set_weights(self.model.get_weights())


    def _build_model(self):
        model = Sequential()
        model.add(Dense(1369, input_dim=self.state_size, activation='relu',kernel_initializer=random_uniform(minval=-0.01,maxval=0.01)))
        model.add(Dense(self.action_size, activation='linear',kernel_initializer=random_uniform(minval=-0.01,maxval=0.01)))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, input, output, priority, result, eprv):
        if len(input) == 0:
            return
        result=2*result-1
        for i in range(len(input)):
            for j in range(len(input[i])):
                reward = 0
                state = input[i][j][0]
                if np.amax(output[i][j]) > 0.9 and priority[i][j] != 0:
                        reward += 0.1

                action = priority[i][j]
                if i == len(input) - 1:
                    reward = result
                    next_state = input[i][j][0]
                    done = True
                else:
                    next_state = input[i + 1][np.argmax(priority[i + 1])][0]
                    done = False
                v = np.argmax(eprv[i][j])
                p = np.power(np.abs(reward - v) + prq_e, 0.5)
                self.memory.add(p,(state, action, reward, next_state, done))

    def randact(self):
        return np.random.randint(self.action_size)

    def act(self, state):
        if np.random.uniform() < self.epsilon and self.test == False:
            return np.random.randint(self.action_size)
        state = np.asarray(state)
        act_value = self.model.predict(state)
        return np.argmax(act_value[0])

    def predict(self,state):
        state = np.asarray(state)
        act_value = self.model.predict(state)
        return act_value[0]

    def replay(self, batch_size):

        self.count += 1
        if self.count == self.max_count:
            self.model.save_weights(weight_file)
            self.copy_weight()
            self.count = 0

        batch=[]
        index = np.random.rand(batch_size) * self.memory.total()
        for i in index:
             batch.append(self.memory.get(i)[2])
        batch=np.asarray(batch)
        #        state, action, reward, next_state, done = batch
        state = np.asarray(list(batch[:, 0]))
        state = np.reshape(state, (state.shape[0], -1))
        action = np.asarray(batch[:, 1], dtype=np.int32)
        reward = np.asarray(batch[:, 2])
        next_state = np.asarray(list(batch[:, 3]))
        done = np.asarray(batch[:, 4])

        q_target = self.model.predict(state)
        q_eval4next = self.model.predict(next_state) #next state ,for max index,,use real time model
        q_next=self.model_target.predict(next_state) #next state ,for value,, use old model
        maxidx=np.argmax(q_eval4next,axis=1)
        index = np.arange(q_next.shape[0])
        target = reward + self.gamma * q_next[index,maxidx]
        for i in range(done.shape[0]):
            if done[i]:
                target[i] = reward[i]
        q_target[index, action] = target  # action and index must be in the form of int32

        loss = self.model.fit(state, q_target, batch_size=batch_size, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = 0
        return loss.history['loss'][0]

    def load(self):
        self.model.load_weights(weight_file)
        self.test = True

    def copy_weight(self):
        self.model_target.set_weights(self.model.get_weights())

# !/usr/bin/env
# coding=utf-8

from keras.layers import Dense
from keras.models import Sequential
import pickle

class DNN:
    def __init__(self, **kwargs):
        self.input_size = kwargs.get("input")
        self.output_size = kwargs.get("output")
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(1369, input_dim=self.input_size, activation='relu'))
        model.add(Dense(self.output_size, activation='linear'))
        return model

    def predict(self, value):
        return self.model.predict(value)

    def loadweight(self, weightfile):
        with open(weightfile,"rb") as f:
            weights=pickle.load(f, encoding='iso-8859-1')
        self.model.set_weights(weights)
        #self.model.load_weights(weightfile)

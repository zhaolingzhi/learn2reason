#coding=utf-8
"""
Activation Layer
setting a and b
let x[x[i]<a] == 0 and x[x[i]>b]==1
"""
import keras
from keras import backend as K
from keras.engine.topology import Layer

class BinaryLayer(Layer):
    def __init__(self, output_dim,a=0.5,b=0.5, **kwargs):
        self.output_dim = output_dim
        self.a=a
        self.b=b
        super(BinaryLayer, self).__init__(**kwargs)

    def call(self, x):
        x[x < self.a] = 0
        x[x >= self.b] = 1
        return

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
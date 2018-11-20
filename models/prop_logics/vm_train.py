#! /usr/bin/env python
#coding=utf-8
import pickle

import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import to_categorical

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
leaves = ['0', '1', 'and', 'or', 'not', '(', ')', \
          '0 and', '0 or', '0 )', '1 and', '1 or', '1 )', \
          'and 0', 'and 1', 'and not', 'and (', \
          'or 0', 'or 1', 'or not', 'or (', \
          'not 0', 'not 1', 'not not', 'not (', \
          '( 0', '( 1', '( not', '( (', \
          ') and', ') or', ') )', 'error!', \
          '( 0 and', '( 0 or', '( 1 and', '( 1 or']

tdate = "curriculum_vm.pkl"
modelf = "vm_model.h5"
lr=0.1
decay=0.001

def createorder(trainxx):
    if len(trainxx)==1:
        return []
    else:
        # tmpo = []
        # lens = len(trainxx)
        # facts = int(factorial(lens - 1))
        # tmpoo = [0 for i in range(lens - 1)]
        # tmpo.append(tmpoo)
        # level = range(lens - 2, -1, -1)
        # for i in range(1, facts):
        #     tmpooo=tmpoo[:]
        #     for j in range(lens - 2,-1,-1):
        #         if tmpooo[j] < level[j]:
        #             tmpooo[j] += 1
        #             break
        #         else:
        #             tmpooo[j] = 0
        #     tmpoo=tmpooo[:]
        #     tmpo.append(tmpoo)
        tmpo=[[0 for i in range(len(trainxx)-1)]]
        return tmpo

def contact(x,y,z):
    con=[]
    if len(x)!=0:
        con = x[:]
    con.append(y)
    if len(z)!=0:
        con += z[:]
    return con

def createmodel():
    input = Input(shape=(74,))
    hidden = Dense(1369, activation='relu')(input)
    output = Dense(37,activation='softmax')(hidden)
    model = Model(input, output)
    return model

def trainmodel(model,x,y):
    model.compile(optimizer=SGD(lr=lr, decay=decay),loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x, y, batch_size=2, epochs=100, verbose=2, callbacks=[
        EarlyStopping(monitor='loss', patience=10, verbose=1, mode='auto'),
        ModelCheckpoint(modelf, monitor='loss', verbose=0, save_best_only=True, mode='auto')])

def predictmodel(model,x):
    result= model.predict(x)
    maxr=max(result[0])
    result[result >= maxr] = 1
    result[result < maxr] = 0
    return list(result[0])

if __name__ == "__main__":
    with open(tdate, 'rb') as f:
        tr = pickle.load(f)
    tx, ty1, ty2 = tr[:3]

    trainx = []
    order = []
    for i in range(len(tx)):
        tmpt = []
        tmpo = []
        for j in range(len(tx[i])):
            tmpt.append(one_hots[tx[i][j]])

        trainx.append(tmpt)
    trainy = to_categorical(ty1, num_classes=37)

    trainxx = []
    trainyy = []
    trainxxl = []
    trainyyl = []
    for j in range(len(trainx)):
        if len(trainx[j]) == 2:
            trainorder = createorder(trainx[j])
            trainxxl.append(trainx[j][0] + trainx[j][1])
            trainyyl.append(trainy[j])
    trainxx = np.asarray(trainxxl)
    trainyy = np.asarray(trainyyl)
    model = createmodel()
    model.load_weights("vm_model.h5")
    trainmodel(model, trainxx, trainyy)

    maxlen = 0
    for x in trainx:
        if len(x) > maxlen:
            maxlen = len(x)

    for i in range(3, maxlen + 1):
        for j in range(len(trainx)):
            if len(trainx[j]) == i:
                trainorder = createorder(trainx[j])
                for line in trainorder:
                    trainxcopy = trainx[j][:]
                    for k in range(len(line)-1):
                        pos = line[k]
                        middlex = np.asarray([trainxcopy[pos] + trainxcopy[pos + 1]])
                        tmp = predictmodel(model, middlex)
                        trainxcopy = contact(trainxcopy[:pos], tmp, trainxcopy[pos + 2:])
                    trainxxl.append(trainxcopy[0] + trainxcopy[1])
                    trainyyl.append(trainy[j])
        trainxx = np.asarray(trainxxl)
        trainyy = np.asarray(trainyyl)
        trainmodel(model, trainxx, trainyy)

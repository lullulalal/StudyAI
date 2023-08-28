import torch as t
from data import ChallengeDataset
from torch.utils.data import WeightedRandomSampler
from Sampler import CustomSampler
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import random
import math

def data_split(dataset):
    datalist = dataset.values.tolist()
    c00 = 0
    c01 = 0
    c10 = 0
    c11 = 0
    list00 = []
    list01 = []
    list10 = []
    list11 = []

    for data in datalist:
        crack = data[1]
        inact = data[2]

        if crack == 0 and inact == 0:
            c00 += 1
            list00.append(data)
        elif crack == 1 and inact == 0:
            c10 += 1
            list10.append(data)
        elif crack == 0 and inact == 1:
            c01 += 1
            list01.append(data)
        else:
            c11 += 1
            list11.append(data)

    random.shuffle(list00)
    random.shuffle(list01)
    random.shuffle(list10)
    random.shuffle(list11)

    trainnum00 = 0
    trainnum01 = 0
    trainnum10 = 0
    trainnum11 = 0

    idx00 = int(round(c00 * 0.8))
    idx01 = int(round(c01 * 0.8))
    idx10 = int(round(c10 * 0.8))
    idx11 = int(round(c11 * 0.8))

    testnum00 = idx00
    testnum01 = idx01
    testnum10 = idx10
    testnum11 = idx11

    for i in range(1):
        trainlist = list00[:idx00]
        trainnum00 += idx00
    for i in range(1):
        trainlist += list01[:idx01]
        trainnum01 += idx01
    for i in range(1):
        trainlist += list10[:idx10]
        trainnum10 += idx10
    for i in range(1):
        trainlist += list11[:idx11]
        trainnum11 += idx11

    testlist = list00[testnum00:]
    testlist += list01[testnum01:]
    testlist += list10[testnum10:]
    testlist += list11[testnum11:]

    random.shuffle(trainlist)
    random.shuffle(testlist)

    train = pd.DataFrame(trainlist,columns = ['filename', 'crack', 'inactive'])
    test = pd.DataFrame(testlist,columns = ['filename', 'crack', 'inactive'])

    print(f"trainnum00:: {trainnum00}, trainnum01:: {trainnum01}")
    print(f"trainnum10:: {trainnum10}, trainnum11:: {trainnum11}")

    q00 = np.ones((trainnum00,2),dtype=int)
    q01 = np.ones((trainnum01,2),dtype=int)
    q10 = np.ones((trainnum10,2),dtype=int)
    q11 = np.ones((trainnum11,2),dtype=int)
    qidx = [0, 0, 0, 0]
    idx = 0
    for data in trainlist:
        if data[1]==0 and data[2]==0:
            q00[qidx[0],1] = int(idx)
            qidx[0] += 1
        elif data[1]==0 and data[2]==1:
            q01[qidx[1],1] = int(idx)
            qidx[1] += 1
        elif data[1]==1 and data[2]==0:
            q10[qidx[2],1] = int(idx)
            qidx[2] += 1
        elif data[1]==1 and data[2]==1:
            q11[qidx[3],1] = int(idx)
            qidx[3] += 1
        idx += 1

    qs = [q00, q01, q10, q11]
    ct = [trainnum00, trainnum01, trainnum10, trainnum11]

    return train, test, qs, ct

data = pd.read_csv('./data.csv', sep=';')
train_data, test_data, qs, ct = data_split(data)

'''
class_counts = train_data['crack'].value_counts().to_list()
num_samples = sum(class_counts)
labels = train_data['crack'].to_list()
class_weights = [num_samples / class_counts[i] for i in range(len(class_counts))]
class_weights = [class_weights[0], class_weights[1]]
weights = [class_weights[labels[i]] for i in range(int(num_samples))]

class_counts2 = train_data['inactive'].value_counts().to_list()
num_samples2 = sum(class_counts2)
labels2 = train_data['inactive'].to_list()
class_weights2 = [num_samples2 / class_counts2[i] for i in range(len(class_counts2))]
class_weights2 = [class_weights2[0], class_weights2[1] / 5]
weights2 = [class_weights2[labels2[i]] for i in range(int(num_samples2))]

weights0 = [(class_weights2[labels2[i]] + class_weights[labels[i]]) / 2 for i in range(int(num_samples2))]
sampler = ImbalancedDatasetSampler(labels)
'''

#sampler = CustomSampler(qs, (45, 1, 15,7))
sample_weight = np.zeros(np.sum(ct))

for i in range(ct[0]):
    idx = qs[0][i,1]
    sample_weight[idx] = 1./ct[0]
for i in range(ct[1]):
    idx = qs[1][i,1]
    sample_weight[idx] = 1./ct[1]
for i in range(ct[2]):
    idx = qs[2][i,1]
    sample_weight[idx] = 1./ct[2]
for i in range(ct[3]):
    idx = qs[3][i,1]
    sample_weight[idx] = 1./ct[3]

samples_weight=t.from_numpy(sample_weight)
sampler = WeightedRandomSampler(samples_weight, len(samples_weight) ,replacement=False)

val_dl = t.utils.data.DataLoader(ChallengeDataset(test_data, 'val'), batch_size=32, shuffle=True)
train_dl = t.utils.data.DataLoader(ChallengeDataset(train_data, 'train'), batch_size=32, shuffle=False, sampler = sampler)
#train_dl = t.utils.data.DataLoader(ChallengeDataset(train_data, 'train'), batch_size=32, shuffle=True)

for batch, (X, y) in enumerate(train_dl):
    X, y = X.to('cpu'), y.to('cpu')

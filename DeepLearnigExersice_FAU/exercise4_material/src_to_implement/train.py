import torch as t
from data import ChallengeDataset
from torch.utils.data import WeightedRandomSampler
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import random
import math
from Sampler import CustomSampler

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

    idx00 = int(round(c00 * 0.75))
    idx01 = int(round(c01 * 0.8))
    idx10 = int(round(c10 * 0.75))
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

res = None
learning_rates = [0.0001]
dropout_rates = [0.5]
weight_decays = [0.00001]
momentums = [0.999]
patience = [7]
batch_size = 32

for mul in range(100):
    train_data, test_data, qs, ct = data_split(data)

    sampler = CustomSampler(qs,(21.5, 0.5, 5.5, 4.5), 'a')

    val_dl = t.utils.data.DataLoader(ChallengeDataset(test_data, 'val'), batch_size=batch_size, shuffle = False)
    train_dl = t.utils.data.DataLoader(ChallengeDataset(train_data, 'train'), batch_size=batch_size, shuffle = True)
    #train_dl = t.utils.data.DataLoader(ChallengeDataset(train_data, 'train'), batch_size=batch_size, shuffle = False, sampler = sampler)

    for patience_i in patience:
        for weight_decay in weight_decays:
            for dropout_rate in dropout_rates:
                for learning_rate_i in learning_rates:
                    for momentum in momentums:
                        #mmodel = model.ResNet()
                        #mmodel = model.DenseNet()
                        #mmodel = model.ResNext()
                        #mmodel = model.WideResNet()
                        mmodel = model.CustomModel()
                        trainer = Trainer(mmodel,
                                          t.nn.BCELoss(),
                                          t.optim.Adam(mmodel.parameters(), lr=learning_rate_i, betas=(0.9, momentum),
                                                       eps=1e-08, weight_decay=weight_decay),
                                          train_dl,
                                          val_dl,
                                          #cuda=True,
                                          cuda=False,
                                          early_stopping_patience=patience_i, loss_weight = None)

                        #if os.path.isfile('checkpoints/checkpoint_{:03d}.ckp'.format(0)):
                            ##trainer.restore_checkpoint(0)

                        print(f"learning_rate_i:: {learning_rate_i:>7f}")
                        res = trainer.fit()
                        print("")

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')

def test_func():
    test_loss = [0.3, 0.2, 0.4, 0.2, 0.4, 0.2, 0.4, 0.2, 0.4, 0.4, 0.3, 0.2, 0.4, 0.2, 0.4, 0.2, 0.4, 0.2, 0.4, 0.4,
                 0.2, 0.2, 0.4, 0.2, 0.4, 0.2, 0.4, 0.2, 0.4, 0.2]
    train_loss = [0.3, 0.2, 0.4, 0.2, 0.4, 0.2, 0.4, 0.2, 0.4, 0.4, 0.3, 0.2, 0.4, 0.2, 0.4, 0.2, 0.4, 0.2, 0.4, 0.4,
                  0.3, 0.2, 0.4, 0.2, 0.4, 0.2, 0.4, 0.2, 0.4, 0.4]
    epoch_cnt = 19
    return train_loss, test_loss, epoch_cnt - 1

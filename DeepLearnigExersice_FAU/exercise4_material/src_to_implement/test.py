import torch as t
import model
from trainer import Trainer
import os
from sklearn.model_selection import train_test_split
import pandas as pd
from data import ChallengeDataset
from sklearn.metrics import precision_score, recall_score, f1_score
import copy
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
import numpy as np

def test():
    data = pd.read_csv('./data.csv', sep=';')
    train_data, test_data = train_test_split(data, test_size=0.9)
    # set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
    # TODO
    val_dl = t.utils.data.DataLoader(ChallengeDataset(test_data, 'val'), batch_size=32)
    train_dl = t.utils.data.DataLoader(ChallengeDataset(train_data, 'train'), batch_size=32)

    resmodel = model.ResNet()

    trainer = Trainer(resmodel,
                      t.nn.BCELoss(),
                      t.optim.Adam(resmodel.parameters(), lr=0.0001, betas=(0.9, 0.999),
                                   eps=1e-08, weight_decay=1e-6),
                      train_dl,
                      val_dl,
                      cuda=True,
                      early_stopping_patience=4)

    #if os.path.isfile('checkpoints/checkpoint_{:03d}.ckp'.format(500)):
    trainer.restore_checkpoint(14)
    #    if learning_rate_i != 0.0002 and learning_rate_i != 0.0001 and learning_rate_i != 0.0003:
    #        os.remove('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_num))

    test_model(trainer._model, val_dl, trainer.device)

def test_model(model, dataloaders, device):
    data = pd.read_csv('./data.csv', sep=';')
    train_data, test_data = train_test_split(data, test_size=0.99)
    # set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
    # TODO
    val_dl = t.utils.data.DataLoader(ChallengeDataset(test_data, 'val'), batch_size=1980)
    train_dl = t.utils.data.DataLoader(ChallengeDataset(train_data, 'train'), batch_size=16)
    dataloaders = val_dl
    model.eval()

    with t.no_grad():
        preds = None
        labels = None
        for i in range(10):
            for data in dataloaders:
                images, y = data
                images = images.to(device)

                pred = model(images)  # file_name

                if preds is None:
                    preds = pred.cpu().data
                else:
                    preds = np.vstack((preds, pred.cpu().data))
                if labels is None:
                    labels = y.detach().cpu().data
                else:
                    labels = np.vstack((labels, y.detach().cpu().data))

                preds[preds < 0.5] = 0
                preds[preds >= 0.5] = 1

        report = classification_report(labels, preds)
        print(report)

test()
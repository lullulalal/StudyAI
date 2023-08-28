import torch as t
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

file_num = 0

class Trainer:

    def __init__(self,
                 model,  # Model to be trained.
                 crit,  # Loss function
                 optim=None,  # Optimizer
                 train_dl=None,  # Training data set
                 val_test_dl=None,  # Validation (or test) data set
                 cuda=True,  # Whether to use the GPU
                 early_stopping_patience=-1, # The patience for early stopping
                 loss_weight=None):
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda
        self.loss_weight = loss_weight
        self._early_stopping_patience = early_stopping_patience
        self.device = 'cpu'

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()
            self.device = 'cuda'
            if loss_weight is not None:
                self.loss_weight[0] = loss_weight[0].cuda()
                self.loss_weight[1] = loss_weight[1].cuda()

    def save_checkpoint(self, epoch):
        #if os.path.isfile('checkpoints/checkpoint_{:03d}.ckp'.format(epoch-5)):
            #os.remove('checkpoints/checkpoint_{:03d}.ckp'.format(epoch-5))
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))


    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      fn,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}})

    def train_step(self, x, y):
        pred = self._model(x)

        #loss = self._crit(pred, y)
        loss1 = self._crit(pred.T[0], y.T[0])
        loss2 = self._crit(pred.T[1], y.T[1])

        if self.loss_weight is not None:
            weight1_ = self.loss_weight[0][y.T[0].data.view(-1).long()].view_as(y.T[0])
            loss_class_weighted1 = loss1 *  weight1_
            loss1 = loss_class_weighted1.mean()
            weight2_ = self.loss_weight[1][y.T[1].data.view(-1).long()].view_as(y.T[1])
            loss_class_weighted2 = loss2 * weight2_
            loss2 = loss_class_weighted2.mean()

        loss = loss1 + loss2

        self._optim.zero_grad()
        loss.backward()
        self._optim.step()

        return loss

    def val_test_step(self, x, y):
        pred = self._model(x)
        loss1 = self._crit(pred.T[0], y.T[0])
        loss2 = self._crit(pred.T[1], y.T[1])
        #loss = self._crit(pred, y)

        if self.loss_weight is not None:
            weight1_ = self.loss_weight[0][y.T[0].data.view(-1).long()].view_as(y.T[0])
            loss_class_weighted1 = loss1 *  weight1_
            loss1 = loss_class_weighted1.mean()
            weight2_ = self.loss_weight[1][y.T[1].data.view(-1).long()].view_as(y.T[1])
            loss_class_weighted2 = loss2 * weight2_
            loss2 = loss_class_weighted2.mean()

        loss = loss1 + loss2

        return pred, loss

    def train_epoch(self):
        self._model.train()
        num_batches = len(self._train_dl)
        train_loss = 0

        for batch, (X, y) in enumerate(self._train_dl):
            X, y = X.to(self.device), y.to(self.device)

            loss = self.train_step(X, y)
            train_loss += loss.item()

        train_loss /= num_batches
        print(f"train_loss: {train_loss:>7f}")
        return train_loss

    def val_test(self):
        self._model.eval()
        num_batches = len(self._val_test_dl)
        test_loss, correct = 0, 0

        preds = None
        labels = None
        with t.no_grad():
            for X, y in self._val_test_dl:
                X, y = X.to(self.device), y.to(self.device)
                pred, loss = self.val_test_step(X, y)
                test_loss += loss.item()

                if preds is None: preds = pred.cpu().data
                else: preds = np.vstack((preds, pred.cpu().data))
                if labels is None: labels = y.detach().cpu().data
                else: labels = np.vstack((labels, y.detach().cpu().data))

                preds[preds < 0.5] = 0
                preds[preds >= 0.5] = 1

        test_loss /= num_batches
        print(f"test_loss: {test_loss:>7f}")
        return test_loss, preds, labels

    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0

        train_loss = []
        test_loss = []
        epoch_cnt = 0

        early_stopper = self.EarlyStopper(self._early_stopping_patience)
        global file_num

        while True:
            train_loss.append(self.train_epoch())
            loss, preds, labels = self.val_test()
            test_loss.append(loss)
            epoch_cnt += 1

            if epochs != -1 and epoch_cnt == epochs:
                break
            if self._early_stopping_patience != -1 and early_stopper.early_stop(loss):
                break

        target_names = ['class 0', 'class 1']
        report = classification_report(labels, preds, target_names=target_names)
        print(report)

        report = classification_report(labels, preds, target_names=target_names, output_dict=True)
        f11 = report['class 0']['f1-score']
        f12 = report['class 1']['f1-score']
        p11 = report['class 0']['precision']
        p12 = report['class 1']['precision']
        r11 = report['class 0']['recall']
        r12 = report['class 1']['recall']

        if (f11 + f12) / 2 >= 0.88 and p11 >=0.85 and p12 >= 0.85 and r11 >= 0.85 and r12 >= 0.85 and p11 != 1.0 and p12 != 1.0 and r11 != 1.0 and r12 != 1.0:
            with open("output.txt", "a") as text_file:
                print(report, file=text_file)

            self.save_checkpoint(file_num+10)
            file_num += 1

        return train_loss, test_loss

    class EarlyStopper:
        def __init__(self, patience=1, min_delta=0):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.min_validation_loss = np.inf

        def early_stop(self, validation_loss):
            if validation_loss < self.min_validation_loss:
                self.min_validation_loss = validation_loss
                self.counter = 0
            elif validation_loss > (self.min_validation_loss + self.min_delta):
                self.counter += 1
                if self.counter >= self.patience:
                    return True
            return False

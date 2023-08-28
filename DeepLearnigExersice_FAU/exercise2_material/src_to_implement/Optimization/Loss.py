import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.prediction = None

    def forward(self, prediction_tensor, label_tensor):
        self.prediction = prediction_tensor + np.finfo(float).eps
        return -np.sum(label_tensor * np.log(self.prediction))

    def backward(self, label_tensor):
        return -label_tensor / self.prediction
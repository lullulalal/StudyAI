import numpy as np
from .Base import BaseLayer

class Dropout(BaseLayer):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability
        self.mask = None

    def forward(self, input_tensor):
        if self.testing_phase is True:
            return input_tensor
        else:
            self.mask = (np.random.rand(*input_tensor.shape) < self.probability) / self.probability
            return input_tensor * self.mask

    def backward(self, error_tensor):
        return error_tensor * self.mask

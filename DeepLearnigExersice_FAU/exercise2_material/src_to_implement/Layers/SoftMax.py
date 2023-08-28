import numpy as np
from .Base import BaseLayer

class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.y_hat = None # prediction

    def forward(self, input_tensor):
        input_tensor = input_tensor - np.max(input_tensor)
        self.y_hat = np.divide(np.exp(input_tensor), np.sum(np.exp(input_tensor), axis=1)[:, np.newaxis])

        return self.y_hat

    def backward(self, error_tensor):
        sec_term = np.subtract(error_tensor, np.sum(np.multiply(error_tensor, self.y_hat), axis=1)[:, np.newaxis])
        return self.y_hat * sec_term

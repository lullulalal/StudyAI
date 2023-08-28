import numpy as np
from .Base import BaseLayer

class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()
        self.shape = None

    def forward(self, input_tensor):
        self.shape = input_tensor.shape

        batch_size = self.shape[0]
        flatten_input_size = int(np.prod(self.shape) / batch_size)

        return input_tensor.reshape(batch_size, flatten_input_size)

    def backward(self, error_tensor):
        return error_tensor.reshape(self.shape)

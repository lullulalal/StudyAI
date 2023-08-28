import numpy as np
from .Base import BaseLayer

class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()
        self.mask = None

    def forward(self, input_tensor):
        self.mask = (input_tensor > 0).astype(int)
        return input_tensor * self.mask

    def backward(self, error_tensor):
        return error_tensor * self.mask
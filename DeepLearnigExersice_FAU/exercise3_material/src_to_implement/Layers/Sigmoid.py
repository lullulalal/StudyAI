import numpy as np
from .Base import BaseLayer

class Sigmoid(BaseLayer):
    def __init__(self):
        super().__init__()
        self.out = None

    def forward(self, input_tensor):
        out = 1 / (1 + np.exp(-input_tensor))
        self.out = out
        return out

    def backward(self, output_tensor):
        dx = output_tensor * (1.0 - self.out) * self.out
        return dx
import numpy as np
from .Base import BaseLayer

class TanH(BaseLayer):
    def __init__(self):
        super().__init__()
        self.out = None

    def forward(self, input_tensor):
        out = np.tanh(input_tensor)
        self.out = out
        return out

    def backward(self, output_tensor):
        dx = output_tensor * (1.0 - self.out ** 2)
        return dx
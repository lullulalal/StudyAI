import numpy as np
from .Base import BaseLayer

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.weights = np.random.uniform(0.0, 1.0, (input_size + 1, output_size))
        self.input = None
        self._optimizer = None
        self._gradient_weights = None

    @property
    def optimizer(self):  # getter
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):  # setter
        self._optimizer = value

    @property
    def gradient_weights(self):  # getter
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):  # setter
        self._gradient_weights = value

    def forward(self, input_tensor):
        self.input = np.c_[input_tensor, np.ones(input_tensor.shape[0])]
        return np.dot(self.input, self.weights)

    def backward(self, error_tensor):
        self.gradient_weights = np.dot(self.input.T, error_tensor)

        if not self.optimizer is None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)

        return np.dot(error_tensor, self.weights[:-1].T)

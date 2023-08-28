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
        self.input_size = input_size
        self.output_size = output_size

        self.weights_norm = 0

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
        if input_tensor.ndim == 1:
            self.input = np.append(input_tensor, [1])
        else:
            self.input = np.c_[input_tensor, np.ones(input_tensor.shape[0])]
        if self.optimizer is not None and self.optimizer.regularizer is not None:
            self.weights_norm = self.optimizer.regularizer.norm(self.weights)
        return np.dot(self.input, self.weights)

    def backward(self, error_tensor):
        if error_tensor.ndim == 1:
            self.gradient_weights = np.dot(self.input[:, None], error_tensor[None, :])
        else:
            self.gradient_weights = np.dot(self.input.T, error_tensor)

        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)

        return np.dot(error_tensor, self.weights[:-1].T)

    def initialize(self, weights_initializer, bias_initializer):
        weights = weights_initializer.initialize((self.input_size, self.output_size), self.input_size, self.output_size)
        bias = bias_initializer.initialize((1, self.output_size), 1, self.output_size)

        self.weights = np.vstack((weights, bias))
import copy
import numpy as np
from .Base import BaseLayer
from .Sigmoid import Sigmoid
from .TanH import TanH
from .FullyConnected import FullyConnected

class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.hidden_state = None

        self._optimizer = None
        self._gradient_weights = None
        self._memorize = False
        self.weights_norm = 0

        self.fcl_first = FullyConnected(self.input_size + self.hidden_size, self.hidden_size)
        self.fcl_second = FullyConnected(self.hidden_size, self.output_size)
        self.cells = []
        self.inputs = []

    @property
    def weights(self):  # getter
        return self.fcl_first.weights

    @weights.setter
    def weights(self, value):  # setter
        self.fcl_first.weights = value

    @property
    def optimizer(self):  # getter
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):  # setter
        self._optimizer = value
        self.fcl_first.optimizer = copy.deepcopy(value)
        self.fcl_second.optimizer = copy.deepcopy(value)

    @property
    def gradient_weights(self):  # getter
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):  # setter
        self._gradient_weights = value

    @property
    def memorize(self):  # getter
        return self._memorize

    @memorize.setter
    def memorize(self, value):  # setter
        self._memorize = value

    def initialize(self, weights_initializer, bias_initializer):
        self.fcl_first.initialize(weights_initializer, bias_initializer)
        self.fcl_second.initialize(weights_initializer, bias_initializer)
        return

    def forward(self, input_tensor):
        time, input_size = input_tensor.shape

        # If memorize is false, initialize the hidden state to zero
        if not self.memorize or self.hidden_state is None:
            self.hidden_state = np.zeros(self.hidden_size)

        y = np.empty((time, self.output_size))

        for t in range(time):
            # Concatenate input_tensor and hidden_state
            xn = np.hstack((input_tensor[t, :], self.hidden_state))

            # calculate ht = tanh(xn·Wh)
            tanh = TanH()
            self.hidden_state = tanh.forward(self.fcl_first.forward(xn))
            # calculate yt = σ(ht·Why + by )
            sigmoid = Sigmoid()
            y[t, :] = sigmoid.forward(self.fcl_second.forward(self.hidden_state))

            # Save inputs for backward
            self.inputs.append((self.fcl_first.input, self.fcl_second.input))

            # Save cells for backward
            self.cells.append([])
            l_idx = len(self.cells) - 1
            self.cells[l_idx].append(self.fcl_first)
            self.cells[l_idx].append(tanh)
            self.cells[l_idx].append(self.fcl_second)
            self.cells[l_idx].append(sigmoid)

            if self.optimizer is not None and self.optimizer.regularizer is not None:
                self.weights_norm += self.fcl_first.weights_norm + self.fcl_second.weights_norm

        return y

    def backward(self, error_tensor):
        time = error_tensor.shape[0]
        dx = np.empty((time, self.input_size))
        dh = 0 # gradient with respect to h of next cell
        dw = 0

        for t in reversed(range(time)):
            dy = error_tensor[t]
            l_idx = len(self.cells) - 1
            i_idx = len(self.inputs) - 1

            input_fcl1, input_fcl2 = self.inputs[i_idx]

            dxh = self.cells[l_idx][3].backward(dy)         # sigmoid
            self.cells[l_idx][2].input = input_fcl2
            dxh = self.cells[l_idx][2].backward(dxh)        # fully connected 2
            dxh = self.cells[l_idx][1].backward(dxh + dh)   # tanh
            self.cells[l_idx][0].input = input_fcl1
            dxh = self.cells[l_idx][0].backward(dxh)        # fully connected 1

            # gradient with respect to h
            dh = dxh[self.input_size:]
            # gradient with respect to x
            dx[t] = dxh[:self.input_size]
            # accumulate gradient with respect to weight
            dw += self.cells[l_idx][0].gradient_weights

            # delete processed a cell and inputs
            del self.cells[-1]
            del self.inputs[-1]

        self.gradient_weights = dw
        return dx
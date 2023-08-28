import copy
import numpy as np
from .Base import BaseLayer

class BatchNormalization(BaseLayer):

    def __init__(self, channels):
        super().__init__()
        self.trainable = True
        self.channels = channels
        self.momentum = 0.8
        self.weights = None
        self.bias = None
        self.initialize()
        self.input_shape = None

        self.running_mean = None
        self.running_var = None

        self.xn = None
        self.x = None
        self.mean = None
        self.var = None

        self.optimizer_w = None
        self.optimizer_b = None
        self._optimizer = None
        self._gradient_weights = None
        self._gradient_bias = None
        self.weights_norm = 0

    @property
    def optimizer(self):  # getter
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):  # setter
        self._optimizer = value
        self.optimizer_w = copy.deepcopy(value)
        self.optimizer_b = copy.deepcopy(value)
        #self.optimizer_w.add_regularizer(None)
        #self.optimizer_b.add_regularizer(None)

    @property
    def gradient_weights(self):  # getter
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):  # setter
        self._gradient_weights = value

    @property
    def gradient_bias(self):  # getter
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, value):  # setter
        self._gradient_bias = value

    def initialize(self, weights_initializer = None, bias_initializer = None):
        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape
        input_ndim = input_tensor.ndim

        if input_ndim != 2:
            input_tensor = self.reformat(input_tensor)

        out = self.__forward(input_tensor)
        if self.optimizer is not None and self.optimizer.regularizer is not None:
            self.weights_norm = self.optimizer.regularizer.norm(self.weights)

        if input_ndim != 2:
            out = self.reformat(out)

        return out

    def __forward(self, x):
        if self.testing_phase is True:
            # normalization
            xn = (x - self.running_mean) / np.sqrt(self.running_var + np.finfo(float).eps)

        else: # training phase
            self.mean = np.mean(x, axis=0)
            self.var = np.var(x, axis=0)
            if self.running_mean is None:
                self.running_mean = self.mean
                self.running_var = self.var

            # normalization
            xn = (x - self.mean) / np.sqrt(self.var + np.finfo(float).eps)
            self.x = x
            self.xn = xn

            # calculate running mean and var for test phase
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.var

        return self.weights * xn + self.bias

    def backward(self, output_tensor):
        output_ndim = output_tensor.ndim

        if output_ndim != 2:
            output_tensor = self.reformat(output_tensor)

        gradient_input = self.__backward(output_tensor)

        if output_ndim != 2:
            gradient_input = self.reformat(gradient_input)

        return gradient_input

    def __backward(self, dout):
        self.gradient_weights = np.sum(dout * self.xn, axis=0)
        self.gradient_bias = np.sum(dout, axis=0)

        if self.optimizer is not None:
            self.weights = self.optimizer_w.calculate_update(self.weights, self.gradient_weights)
            self.bias = self.optimizer_b.calculate_update(self.bias, self.gradient_bias)

        return self.__compute_bn_gradients(dout, self.x, self.weights, self.mean, self.var)

    def reformat(self, tensor):
        if tensor.ndim == 2:
            batch_size, channel, height, width = self.input_shape
            tensor = tensor.reshape((batch_size,-1, channel)).transpose((0, 2, 1))
            tensor = tensor.reshape(self.input_shape)
        else:
            batch_size, channel, height, width = tensor.shape
            tensor = tensor.reshape((batch_size, channel, -1)).transpose((0, 2, 1))
            tensor = tensor.reshape((-1, channel))

        return tensor

    def __compute_bn_gradients(self, error_tensor, input_tensor, weights, mean, var, eps=np.finfo(float).eps):
        if eps > 1e-10:
            raise ArithmeticError("Eps must be lower than 1e-10. Your eps values %s" % (str(eps)))

        norm_mean = input_tensor - mean
        var_eps = var + eps

        gamma_err = error_tensor * weights
        inv_batch = 1. / error_tensor.shape[0]

        grad_var = np.sum(norm_mean * gamma_err * -0.5 * (var_eps ** (-3 / 2)), keepdims=True, axis=0)

        sqrt_var = np.sqrt(var_eps)
        first = gamma_err * 1. / sqrt_var

        grad_mu_two = (grad_var * np.sum(-2. * norm_mean, keepdims=True, axis=0)) * inv_batch
        grad_mu_one = np.sum(gamma_err * -1. / sqrt_var, keepdims=True, axis=0)

        second = grad_var * (2. * norm_mean) * inv_batch
        grad_mu = grad_mu_two + grad_mu_one

        return first + second + inv_batch * grad_mu
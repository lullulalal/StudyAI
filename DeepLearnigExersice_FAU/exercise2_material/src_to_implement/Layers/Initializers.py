import numpy as np

class Constant:
    def __init__(self, value = 0.1):
        self.value = value

    def initialize(self, weights_shape, fan_in = 0, fan_out = 0):
        return np.zeros(weights_shape) + self.value

class UniformRandom:
    # noinspection PyMethodMayBeStatic
    def initialize(self, weights_shape, fan_in = 0, fan_out = 0):
        return np.random.uniform(0, 1, weights_shape)

class Xavier:
    # noinspection PyMethodMayBeStatic
    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2 / (fan_in + fan_out))
        return np.random.normal(0, sigma, weights_shape)

class He:
    # noinspection PyMethodMayBeStatic
    def initialize(self, weights_shape, fan_in, fan_out = 0):
        sigma = np.sqrt(2 / fan_in)
        return np.random.normal(0, sigma, weights_shape)
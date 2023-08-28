import copy
import numpy as np
from .Base import BaseLayer

class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True

        self.is_1d = False
        if len(stride_shape) == 1:
            convolution_shape = (convolution_shape[0], 1, convolution_shape[1])
            stride_shape = (1, stride_shape[0])
            self.is_1d = True

        self.weights = np.random.uniform(0.0, 1.0, (num_kernels, *convolution_shape))
        self.bias = np.random.uniform(0.0, 1.0, num_kernels)

        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.input_shape = None
        self.input_column = None
        self.output_shape = None

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

    def forward(self, input_tensor):
        if self.is_1d is True:
            input_tensor = input_tensor.reshape((input_tensor.shape[0], input_tensor.shape[1], 1, input_tensor.shape[2]))

        batch_size, channel, input_h, input_w = input_tensor.shape
        stride_y, stride_x = self.stride_shape
        output_h = input_h // stride_y + int(bool(input_h % stride_y))
        output_w = input_w // stride_x + int(bool(input_w % stride_x))
        self.output_shape = (output_h, output_w)

        input_column = self.image_to_column(input_tensor)
        weights_column = self.weights.reshape((self.num_kernels, -1)).T

        output = np.dot(input_column, weights_column) + self.bias
        output = output.reshape((batch_size, output_h, output_w, -1)).transpose((0, 3, 1, 2))
        if self.is_1d is True:
            output = output.reshape((output.shape[0], output.shape[1], output.shape[3]))

        self.input_shape = input_tensor.shape
        self.input_column = input_column

        if self.optimizer_w is not None and self.optimizer_w.regularizer is not None:
            self.weights_norm = self.optimizer_w.regularizer.norm(self.weights)

        return output

    def backward(self, error_tensor):
        if self.is_1d is True:
            error_tensor = error_tensor.reshape((error_tensor.shape[0], error_tensor.shape[1], 1, error_tensor.shape[2]))

        error_tensor = error_tensor.transpose((0,2,3,1)).reshape((-1, self.num_kernels))

        self.gradient_bias = np.sum(error_tensor, axis=0)
        self.gradient_weights = np.dot(self.input_column.T, error_tensor)
        self.gradient_weights = self.gradient_weights.transpose((1, 0)).reshape(self.weights.shape)

        if self.optimizer is not None:
            self.weights = self.optimizer_w.calculate_update(self.weights, self.gradient_weights)
            self.bias = self.optimizer_b.calculate_update(self.bias, self.gradient_bias)

        weights_column = self.weights.reshape((self.num_kernels, -1))
        gradient_input = np.dot(error_tensor, weights_column)
        #gradient with respect to the input for each window
        gradient_input = self.column_to_image(gradient_input)

        if self.is_1d is True:
            gradient_input = gradient_input.reshape((gradient_input.shape[0], gradient_input.shape[1], gradient_input.shape[3]))

        return gradient_input

    def initialize(self, weights_initializer, bias_initializer):
        fan_in = np.prod(self.convolution_shape)
        fan_out = self.num_kernels * np.prod(self.convolution_shape[1:])
        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape)

    def image_to_column(self, input_tensor):
        batch_size, channel, input_h, input_w = input_tensor.shape
        stride_y, stride_x = self.stride_shape
        filter_h, filter_w = self.convolution_shape[1:]

        pad_h = filter_h - (input_h - ((input_h - 1) // stride_y) * stride_y)
        pad_t = pad_h // 2
        pad_b = pad_t + pad_h % 2
        pad_w = filter_w - (input_w - ((input_w - 1) // stride_x) * stride_x)
        pad_l = pad_w // 2
        pad_r = pad_l + pad_w % 2
        image = np.pad(input_tensor, [(0, 0), (0, 0), (pad_t, pad_b), (pad_l, pad_r)], 'constant')

        output_h, output_w = self.output_shape
        column = np.zeros((batch_size, channel, filter_h, filter_w, output_h, output_w))

        for y in range(filter_h):
            y_end = y + stride_y * output_h
            for x in range(filter_w):
                x_end = x + stride_x * output_w
                column[:, :, y, x, :, :] = image[:, :, y:y_end:stride_y, x:x_end:stride_x]

        column = column.transpose((0, 4, 5, 1, 2, 3)).reshape((batch_size * output_h * output_w, -1))
        return column

    def column_to_image(self, column):
        batch_size, channel, input_h, input_w = self.input_shape
        stride_y, stride_x = self.stride_shape
        filter_h, filter_w = self.convolution_shape[1:]

        output_h, output_w = self.output_shape
        column = column.reshape((batch_size, output_h, output_w, channel, filter_h, filter_w)).transpose((0, 3, 4, 5, 1, 2))

        pad_h = filter_h - (input_h - ((input_h - 1) // stride_y) * stride_y)
        pad_w = filter_w - (input_w - ((input_w - 1) // stride_x) * stride_x)
        image = np.zeros((batch_size, channel, input_h + pad_h, input_w + pad_w))

        for y in range(filter_h):
            y_end = y + stride_y * output_h
            for x in range(filter_w):
                x_end = x + stride_x * output_w
                image[:, :, y:y_end:stride_y, x:x_end:stride_x] += column[:, :, y, x, :, :]

        pad_t = pad_h // 2
        pad_l = pad_w // 2
        return image[:, :, pad_t:input_h + pad_t, pad_l:input_w + pad_l]
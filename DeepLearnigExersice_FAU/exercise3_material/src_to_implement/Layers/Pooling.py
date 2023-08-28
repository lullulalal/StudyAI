import numpy as np
from .Base import BaseLayer

class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.input_shape = None
        self.output_shape = None
        self.index_max = None

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape
        batch_size, channel, input_h, input_w = input_tensor.shape
        pooling_h, pooling_w = self.pooling_shape
        stride_y, stride_x = self.stride_shape
        output_h = (input_h - pooling_h) // stride_y + 1
        output_w = (input_w - pooling_w) // stride_x + 1
        self.output_shape = (output_h, output_w)

        input_column = self.image_to_column(input_tensor)
        self.index_max = np.argmax(input_column, axis=1)
        max_column = np.max(input_column, axis=1)

        return max_column.reshape((batch_size, channel, output_h, output_w))

    def backward(self, error_tensor):
        gradient_column = np.zeros((np.prod(error_tensor.shape), np.prod(self.pooling_shape)))
        gradient_column[np.arange(self.index_max.size), self.index_max] = error_tensor.flatten()

        return self.column_to_image(gradient_column)

    def image_to_column(self, input_tensor):
        batch_size, channel, input_h, input_w = input_tensor.shape
        pooling_h, pooling_w = self.pooling_shape
        stride_y, stride_x = self.stride_shape
        output_h, output_w = self.output_shape

        image = input_tensor
        column = np.zeros((batch_size, channel, pooling_h, pooling_w, output_h, output_w))

        for y in range(pooling_h):
            y_end = y + stride_y * output_h
            for x in range(pooling_w):
                x_end = x + stride_x * output_w
                column[:, :, y, x, :, :] = image[:, :, y:y_end:stride_y, x:x_end:stride_x]

        column = column.transpose((0, 1, 4, 5, 2, 3)).reshape((-1, np.prod(self.pooling_shape)))

        return column

    def column_to_image(self, column):
        batch_size, channel, input_h, input_w = self.input_shape
        pooling_h, pooling_w = self.pooling_shape
        stride_y, stride_x = self.stride_shape
        output_h, output_w = self.output_shape

        column = column.reshape((batch_size, channel, output_h, output_w, pooling_h, pooling_w)).transpose((0, 1, 4, 5, 2, 3))
        image = np.zeros((batch_size, channel, input_h, input_w))

        for y in range(pooling_h):
            y_end = y + stride_y * output_h
            for x in range(pooling_w):
                x_end = x + stride_x * output_w
                image[:, :, y:y_end:stride_y, x:x_end:stride_x] += column[:, :, y, x, :, :]

        return image
import numpy as np
import matplotlib.pyplot as plt

class Checker:
    def __init__(self, resolution, tile_size):
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = None

    def draw(self):
        black_pixel_row = np.zeros(self.tile_size)
        white_pixel_row = np.ones(self.tile_size)

        combine_row_bw = np.concatenate((black_pixel_row, white_pixel_row))
        board_pixel_row_bw = np.tile(combine_row_bw, self.resolution // (self.tile_size * 2))

        combine_row_wb = np.concatenate((white_pixel_row, black_pixel_row))
        board_pixel_row_wb = np.tile(combine_row_wb, self.resolution // (self.tile_size * 2))

        board_tile_row_bw = np.tile(board_pixel_row_bw, (self.tile_size, 1))
        board_tile_row_wb = np.tile(board_pixel_row_wb, (self.tile_size, 1))
        self.output = np.tile(np.vstack((board_tile_row_bw, board_tile_row_wb)), (self.resolution // (self.tile_size * 2), 1))

        return self.output.copy()

    def show(self):
        if self.output is not None:
            plt.imshow(self.output, cmap = 'gray')
            plt.show()

class Circle:
    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = None

    def draw(self):
        line = np.linspace(0, self.resolution - 1, self.resolution)

        x, y = np.meshgrid(line, line)
        f = (x - self.position[0]) ** 2 + (y - self.position[1]) ** 2

        circle = f <= (self.radius ** 2)
        self.output = circle

        return self.output.copy()

    def show(self):
        if self.output is not None:
            plt.imshow(self.output, cmap = 'gray')
            plt.show()

class Spectrum:
    def __init__(self, resolution):
        self.resolution = resolution
        self.output = None

    def draw(self):
        self.output = np.zeros([self.resolution, self.resolution, 3])

        self.output[:, :, 0] = np.linspace(0, 1, self.resolution)
        self.output[:, :, 1] = np.linspace(0, 1, self.resolution).reshape(self.resolution, 1)
        self.output[:, :, 2] = np.linspace(1, 0, self.resolution)

        return self.output.copy()

    def show(self):
        if self.output is not None:
            plt.imshow(self.output, vmin=0, vmax=1)
            plt.show()
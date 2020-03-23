import numpy as np

class ShapesHelper():
    def __init__(self, x, f, stride):
        self.x = x # image
        self.f = f # filter
        self.stride = stride

        self.initialize_shapes()
        self.unique_positions = self.get_unique_positions()

    def initialize_shapes(self):
        self.f_rows, self.f_cols = self.f.shape
        self.x_rows, self.x_cols = self.x.shape

        # only accept the size N=M
        assert self.f_rows == self.f_cols
        assert self.x_rows == self.x_cols

        # matrix coordinates
        self.x_top, self.x_bottom = 0, self.f_cols
        self.y_top, self.y_bottom = 0, self.f_rows

        # only accept stride that gives a whole number
        check_stride_size = (self.x_cols-self.f_cols)/(self.stride)+1
        assert (check_stride_size).is_integer()

    def get_unique_positions(self):
        x_positions = self._get_num_x_positions()
        y_positions = self._get_num_y_positions()

        return x_positions * y_positions

    def get_all_params(self):
        return self.unique_positions, self.x_top, self.x_bottom, self.y_top, self.y_bottom, \
               self.x_rows, self.x_cols, self.f_rows, self.f_cols

    def _get_num_x_positions(self):
        num_positions_x = 0

        for i in range(self.x_cols):
            num_positions_x += 1

            self.x_top += self.stride
            self.x_bottom += self.stride

            if self._check_position():
                break

        return num_positions_x

    def _get_num_y_positions(self):
        num_positions_y = 0

        for i in range(self.x_rows):
            num_positions_y += 1

            self.y_top += self.stride
            self.y_bottom += self.stride

            if self._check_position():
                break

        return num_positions_y

    def _check_position(self):
        position = self.x[self.y_top : self.y_bottom, self.x_top: self.x_bottom]
        if position.size == 0 or position.size != self.f_rows*self.f_cols:
            # reset and stop loop
            self.initialize_shapes()
            return True

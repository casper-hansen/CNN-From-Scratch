import numpy as np

class ConvolutionalNeuralNetwork():

    def __init__(self, x, f, stride, get_all_params):
        self.x = x # image / input
        self.f = f # filter
        self.stride = stride
        
        self.unique_positions, \
            self.x_top, self.x_bottom, \
            self.y_top, self.y_bottom, \
            self.x_rows, self.x_cols, \
            self.f_rows, self.f_cols = get_all_params()

    def calculate_spatial_positions(self):
        # initialize slices
        self.slices = []
        self.slices.append(self.x[self.y_top : self.y_bottom, self.x_top: self.x_bottom])

        for i in range(0, self.unique_positions-1):
            # move left to right
            self.x_top += self.stride
            self.x_bottom += self.stride

            position = self.x[self.y_top : self.y_bottom, self.x_top: self.x_bottom]
            if position.size == 0 or position.size != self.f_rows*self.f_cols:
                # if we are at the right edge, move back left and go downwards
                self.x_top = 0
                self.x_bottom = self.f_cols
                self.y_top += self.stride
                self.y_bottom += self.stride

                # TODO: check if we are at y-axis edge? not possible, because of sizes?
                position = self.x[self.y_top : self.y_bottom, self.x_top: self.x_bottom]
                #print(f'x[{y_top}:{y_bottom}, {x_top}:{x_bottom}]')
            
            self.slices.append(position)

    def calculate_dot_products(self):
        # calculate output size and prepare array
        output_size = int((self.x_cols-self.f_cols)/(self.stride)+1)
        output_dots = np.zeros((output_size, output_size), dtype=int)

        # fill output array
        slice_counter = 0
        for i in range(output_size):
            for j in range(output_size):
                position = self.slices[slice_counter]
                #Same effect: np.sum(np.multiply(position, self.f))
                output_dots[i][j] = np.vdot(position, self.f) # + bias, if you want to add bias
                slice_counter += 1

        print(output_dots)

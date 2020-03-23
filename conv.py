from shapes_helper import ShapesHelper
import numpy as np

x = np.array([[1, 2, 3, 4],
              [2, 1, 6, 6],
	          [1, 5, 4, 3],
              [5, 4, 8, 2]])

f = np.array([[2, 3],
              [1, 1]])

stride = 2

helper = ShapesHelper(x, f, stride)

# get all the parameters
unique_positions, x_top, x_bottom, y_top, y_bottom, \
                  x_rows, x_cols, f_rows, f_cols = helper.get_all_params()

# initialize slices
slices = []
slices.append(x[y_top : y_bottom, x_top: x_bottom])

for i in range(0, unique_positions-1):
    # move left to right
    x_top += stride
    x_bottom += stride

    position = x[y_top : y_bottom, x_top: x_bottom]
    if position.size == 0 or position.size != f_rows*f_cols:
        # if we are at the right edge, move back left and go downwards
        x_top = 0
        x_bottom = f_cols
        y_top += stride
        y_bottom += stride

        #print(f'x[{y_top}:{y_bottom}, {x_top}:{x_bottom}]')

        position = x[y_top : y_bottom, x_top: x_bottom]
    
    slices.append(position)

# calculate output size and fill out array
output_size = int((x_cols-f_cols)/(stride)+1)
out_array = np.zeros((output_size, output_size))
print(out_array)

for position in slices:
    print(position)
    output = np.vdot(position, f)

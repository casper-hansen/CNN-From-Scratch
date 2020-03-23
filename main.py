from shapes_helper import ShapesHelper
from cnn import ConvolutionalNeuralNetwork
import numpy as np

x = np.array([[1, 2, 3, 4],
              [2, 1, 6, 6],
	          [1, 5, 4, 3],
              [5, 4, 8, 2]])

f = np.array([[2, 3],
              [1, 1]])

stride = 2

helper = ShapesHelper(x, f, stride)

cnn = ConvolutionalNeuralNetwork(x, f, stride, helper.get_all_params)

cnn.calculate_spatial_positions()
cnn.calculate_dot_products()
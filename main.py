from cnn import ConvolutionalNeuralNetwork
from layers.conv2d import Conv2D
from layers.dense import Dense
import numpy as np

x = np.array([[1, 2, 3, 4],
              [2, 1, 6, 6],
	          [1, 5, 4, 3],
              [5, 4, 8, 2]])

f = np.array([[2, 3],
              [1, 1]])

stride = 2

cnn = ConvolutionalNeuralNetwork()

cnn.add(Conv2D(f, stride))
cnn.add(Dense())

cnn.fit(x)
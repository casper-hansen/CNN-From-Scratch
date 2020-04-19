import numpy as np

class ConvolutionalNeuralNetwork():
    def __init__(self):
        self.layers = []

    def add(self, layer_to_add):
        self.layers.append(layer_to_add)

    def fit(self):
        for layer in self.layers:
            output = layer.forward()
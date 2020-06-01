import numpy as np
import copy

class ConvolutionalNeuralNetwork():
    def __init__(self):
        self.layers = []

    def add(self, layer_to_add):
        self.layers.append(layer_to_add)

    def forward_pass(self, x):
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def fit(self, x):
        output = self.forward_pass(x)
        print(output)
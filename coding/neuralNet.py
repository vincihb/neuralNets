import numpy as np
import math 
import mnistLoad

class net(object):
    arrayOfNeurons = np.array([]);

    def description(self):
        print(self.arrayOfNeurons)
    
    def initialise(self):
        w, b = initialiseNet(arrayOfNeurons)

    def feedForward(inputArray, weights, biases, neurons):
        activations = inputArray
        i = 0
        while i < len(neurons)-1:
            activations = sigmoidArray((weights[i].dot(activations) + biases[i])[0])
            i = i + 1
            print(activations)
        return activations

    def initialiseNet(array):
        weights = []
        biases = []
        i = 0
        while i < len(array)-1:
            weights.append(np.random.random((array[i+1], array[i])))
            biases.append(np.random.random((array[i+1],1)))
            i = i + 1
        return weights, biases

    def sigmoidArray(array):
        i = 0
        while i < len(array):
            array[i] = sigmoid(array[i])
            i = i + 1
        return array

#misc functions
def sigmoid(x):
    return 1/(1+math.exp(-x))

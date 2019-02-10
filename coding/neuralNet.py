import numpy as np
import math 
import mnistLoad

class net(object):

    zeds = []
    layersNet = 0


    def __init__(self, array):
        self.array = array
    
    def description(self):
        print(self.array)
    
    def feedForward(self, inputArray, weights, biases, neurons):
        activations = inputArray
        i = 0
        while i < len(neurons)-1:
            zeds.append(normalizeArray((weights[i].dot(activations) + biases[i])[0]))
            activations = normalizeArray(sigmoidArray(zeds[i]))
            i = i + 1
        return activations

    def initialiseNet(self, array):
        weights = []
        biases = []
        layersNet = len(array)
        i = 0
        while i < len(array)-1:
            weights.append(np.random.random((array[i+1], array[i])))
            biases.append(np.random.random((array[i+1],1)))
            i = i + 1
        return weights, biases

    def feedBack(self, labelArray, outputArray):
        arrayA = []
        deltaLayers = []
        i = 0
        while i < len(outputArray):
            deltaLayers.append((outputArray[i] - labelArray[i]) 
                    * zeds[i]) 
            
            



#misc functions
def normalizeArray(array):
    i = 0
    sumArray = 0
    while i < len(array):
        sumArray = array[i] + sumArray
        i = i + 1
    i = 0
    while i < len(array):
        array[i] = array[i]/sumArray
        i = i + 1
    return array

def sigmoid(x):
    return 1.0/(1+math.exp(-x))

def sigmoidDerivative(x):
    return math.exp(-x)/((1+math.exp(-x))**2)

def sigmoidArray(array):
    i = 0
    while i < len(array):
        array[i] = sigmoid(array[i])
        i = i + 1
    return array



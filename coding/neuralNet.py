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
        print(self.zeds)
        print(self.layersNet)
    
    def feedForward(self, inputArray):
        activations = inputArray
        i = 0
        while i < self.layersNet - 1:
            self.zeds.append(normalizeArray((self.weights[i].dot(activations) + self.biases[i])[0]))
            activations = normalizeArray(sigmoidArray(self.zeds[i]))
            i = i + 1
        return activations

    def initialiseNet(self, array):
        weights = []
        biases = []
        self.layersNet = len(array)
        i = 0
        while i < len(array)-1:
            weights.append(np.random.random((array[i+1], array[i])))
            biases.append(np.random.random((array[i+1],1)))
            i = i + 1
        self.weights = weights
        self.biases = biases
        return weights, biases

    def feedBack(self, label, outputArray):
        deltaLayers = []
        deltaLayers.append(np.diag(outputArray - createLabelArray(label)).dot(normalizeArray(
                sigmoidDerArray(self.zeds[len(self.zeds) - 1]))))
        
        i = 0
        while i < self.layersNet:
            deltaLayers.append(np.diag(np.transpose(self.weights[self.layersNet-1-i]).dot(
                deltaLayers[i])).dot(normalizeArray(sigmoidDerArray(self.zeds[]
                        )
                    )
                )
            )
            i = i + 1
        #   deltaLayers.append(np.absolute((self.weights[self.layersNet - i - 1] * deltaLayers[i]) * 
        #        sigmoidDerArray(self.zeds[self.layersNet - i - 1])))
        #    i = i + 1
        #i = 0
        #while i < self.layersNet:
        #    self.biases[self.layersNet - i - 1] = (self.biases[self.layersNet - i - 1] 
        #            - deltaLayers[i])
        #    i = i + 1

        return 1 

                    
            



#misc functions
def createLabelArray(label):
    array = []
    i = 0
    while i < 10:
        array.append(0)
        i = i + 1
    array[label] = 1
    return array

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

def sigmoidDerArray(array):
    i = 0
    while i < len(array):
        array[i] = sigmoidDerivative(array[i])
        i = i + 1
    return array

def sigmoidArray(array):
    i = 0
    while i < len(array):
        array[i] = sigmoid(array[i])
        i = i + 1
    return array



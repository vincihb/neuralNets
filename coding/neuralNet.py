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
    
    def feedForward(self, inputArray, flag):
        print self.weights[0][0][0]
        activations = inputArray
        if flag == 0:
            self.zeds.append(inputArray)
            i = 0
            while i < self.layersNet - 1:
                self.zeds.append(normalizeArray((self.weights[i].dot(activations) 
                    + self.biases[i])[0]))
                activations = normalizeArray(sigmoidArray(self.zeds[i+1]))
                i = i + 1
        else:
            self.zeds[0] = inputArray
            i = 0
            while i < self.layersNet - 1:
                self.zeds[i+1] = (normalizeArray((self.weights[i].dot(activations) 
                    + self.biases[i])[0]))
                activations = normalizeArray(sigmoidArray(self.zeds[i+1]))
                i = i + 1
        return activations

    def initialiseNet(self):
        weights = []
        biases = []
        self.layersNet = len(self.array)
        i = 0
        while i < len(self.array)-1:
            weights.append(np.random.random((self.array[i+1], self.array[i])))
            biases.append(np.random.random((self.array[i+1],1)))
            i = i + 1
        self.weights = rowStochMatrix(weights)
        self.biases = rowStochMatrix(biases)
        return self.weights, self.biases

    def feedBack(self, label, outputArray):
        deltaLayers = []
        deltaLayers.append(np.diag(outputArray - createLabelArray(label)).dot(
            normalizeArray(sigmoidDerArray(self.zeds[len(self.zeds) - 1]))))    
        print createLabelArray(label)
        i = 0
        while i < self.layersNet - 1:
            deltaLayers.append(np.diag(np.transpose(self.weights[self.layersNet-2-i]).dot(
                deltaLayers[i])).dot(normalizeArray(sigmoidDerArray(self.zeds[self.layersNet - 2 - i]
                        )
                    )
                )
            )
            i = i + 1
        weightDiff = []
        i = 0 
        while i < self.layersNet - 1:
            weightDiff.append(rowStochMatrix(matrixMult(deltaLayers[self.layersNet - i - 1], 
                normalizeArray(sigmoidArray(self.zeds[i+1])))))
            i = i + 1
        i = 0
        nabla = 10000000
        while i < self.layersNet - 1:
            self.weights[i] = rowStochMatrix(nabla*(self.weights[i] - np.transpose(weightDiff[i])))
            self.biases[self.layersNet - i - 2] = normalizeArray(nabla*(self.biases[
                self.layersNet - i - 2] 
                    - deltaLayers[i]))
            i = i + 1
        return 1 

#misc functions
def rowStochMatrix(arrayM):
    i = 0
    while i < len(arrayM):
        arrayM[i] = normalizeArray(arrayM[i])
        i = i + 1
    return arrayM


def matrixMult(array1, array2):
    array = []
    i = 0 
    while i < len(array1):
        j = 0
        array.append([])
        while j < len(array2):
            array[i].append(array1[i]*array2[j])
            j = j + 1
        i = i + 1
    #print "(" + str(i) + "," + str(j) 
    return array
    

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



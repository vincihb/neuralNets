import numpy as np
import math 
import mnistLoad
import vector as vc

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
        activations = vc.Vector(inputArray)
        tempRet = vc.Vector(inputArray)
        if flag == 0:
            self.zeds.append(activations)
            i = 0
            while i < self.layersNet - 1:
                tempRet.matrixProduct(self.weights[i])
                tempRet.vecAdd(self.biases[i])
                tempRet.normalizeVector()
                self.zeds.append(tempRet)
                activations = tempRet.sigmoidArray()
                i = i + 1
        else:
            self.zeds[0] = vc.Vector(inputArray)
            i = 0
            while i < self.layersNet - 1:
                tempRet.matrixProduct(self.weights[i])
                tempRet.vecAdd(self.biases[i])
                tempRet.normalizeVector()
                self.zeds[i+1] = tempRet
                activations = tempRet.sigmoidArray()
                #self.zeds[i+1] = (normalizeArray(arrayAdd(self.weights[i].dot(activations),
                #    self.biases[i])))
                #activations = normalizeArray(sigmoidArray(self.zeds[i+1]))
                i = i + 1
        return activations

    def initialiseNet(self):
        weights = []
        biases = []
        self.layersNet = len(self.array)
        i = 0
        while i < len(self.array)-1:
            weights.append(rowStochMatrix(np.random.random((self.array[i+1], self.array[i]))))
            biases.append(normalizeArray(np.random.random((self.array[i+1],1))))
            i = i + 1
        self.weights = weights
        self.biases = biases
        return self.weights, self.biases

    def feedBack(self, label, outputArray):
        deltaLayers = []
        outputA = vc.Vector(outputArray)
        labelA = vc.Vector(createLabelArray(label))
        outputA.vecDiff(labelA.array)
        tempRet = self.zeds[len(self.zeds) - 1]
        #tempRet = self.zeds[len(self.zeds) - 1]
        tempRet.sigmoidDerArray()
        tempRet.matrixProduct(np.diag(outputA.array))
        deltaLayers.append(tempRet)
        #deltaLayers.append(np.diag(arrayDiff(outputArray, createLabelArray(label))).dot(
        #    normalizeArray(sigmoidDerArray(self.zeds[len(self.zeds) - 1]))))    
        i = 0
        while i < self.layersNet - 1:
            tempRet = self.zeds[len(self.zeds) - 2 - i]
            #tempRet = self.zeds[len(self.zeds) -2 - i]
            tempRet.sigmoidDerArray()
            prevDelta = deltaLayers[i]
            prevDelta.matrixProduct(np.transpose(self.weights[self.layersNet - 2 - i]))
            tempRet.matrixProduct(np.diag(prevDelta.array))
            deltaLayers.append(tempRet)
            #deltaLayers.append(np.diag(np.transpose(self.weights[self.layersNet-2-i]).dot(
            #    deltaLayers[i])).dot(normalizeArray(sigmoidDerArray(self.zeds[self.layersNet - 2 - i]
            #            )
            #        )
            #    )
            #)
            i = i + 1
        weightDiff = []
        i = 0 
        while i < self.layersNet - 1:
            zedCall = self.zeds[i + 1]
            delCall = deltaLayers[self.layersNet - i - 1]
            weightDiff.append(rowStochMatrix(delCall.matrixMult(zedCall.sigmoidArray())))
            #weightDiff.append(rowStochMatrix(matrixMult(deltaLayers[self.layersNet - i - 1], 
            #    normalizeArray(sigmoidArray(self.zeds[i+1])))))
            i = i + 1
        i = 0
        #print weightDiff[0][0][0]
        while i < self.layersNet - 1:
            self.weights[i] = rowStochMatrix(matrixDiff(self.weights[i], np.transpose(weightDiff[i])))
            #self.weights[i] = rowStochMatrix(arrayDiff(self.weights[i],
            #    np.transpose(weightDiff[i])))
            tempB = vc.Vector(self.biases[self.layersNet - i - 2])
            tempB.vecDiff(deltaLayers[i].array)
            tempB.normalizeVector()
            self.biases[self.layersNet - i - 2] = tempB.array
            #self.biases[self.layersNet - i - 2] = normalizeArray((self.biases[self.layersNet - i - 2]
            #    - deltaLayers[i].array))
            i = i + 1
        return 1 

#misc functions
def matrixDiff(matrix1, matrix2):
    matrixRet = []
    i = 0
    while i < len(matrix1):
        j = 0
        arrayR = []
        while j < len(matrix1[0]):
            arrayR.append(abs(matrix1[i][j] - matrix2[i][j]))
            j = j + 1
        matrixRet.append(arrayR)
        i = i + 1
    return matrixRet

def arrayDiff(array1, array2):
    array = []
    i = 0 
    while i < len(array1):
        array.append(array1[i] - array2[i])
        i = i + 1
    return array

def arrayAdd(array1, array2):
    array = []
    i = 0
    while i < len(array1):
        array.append(array1[i] + array2[i])
        i = i + 1
    return array

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



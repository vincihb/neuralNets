import numpy as np
import math 
import mnistLoad
import vector as vc

class net(object):

    zeds = []
    layersNet = 0

    #Initialise the neural network
    def __init__(self, array):
        self.array = array
        self.learningRates = []
    
    #Nifty little function to describe what the attributes of the neural network is
    def description(self):
        print(self.array)
        print(self.zeds)
        print(self.layersNet)
        print(self.learningRates)
    
    #Feedforward loop
    def feedForward(self, inputArray, flag):
        print self.weights[0][0][0] #For debugging, printing out what the weight is 
        activations = vc.Vector(inputArray) #Create a vector for the activations 
        tempRet = vc.Vector(inputArray) #Create a temporary vector for the input array
        if flag == 0: #Check if this is the first training loop
            self.zeds.append(activations) #Add the activations of the first layer as zeds
            i = 0
            while i < self.layersNet - 1:
                tempRet.matrixProduct(self.weights[i]) #Multiply the weights with the current activations
                tempRet.vecAdd(self.biases[i]) #Add the biases
                tempRet.normalizeVector() #Normalize
                self.zeds.append(tempRet) #And we get the zeds of layer i ((i+1)th layer)
                activations = tempRet.sigmoidArray() #Activations are just the sigmoid of the zeds we created
                i = i + 1 #Iterate to go to the next layer
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

    #Aside from the other initialisation function, we have a seperate function for initialising the weights
    #biases, and the moment estimates for the stochastic gradient descent algorithm
    def initialiseNet(self):
        weights = []
        biases = []
        m1Biases = []
        m2Biases = []
        m1Weights = []
        m2Weights = []
        self.layersNet = len(self.array)
        i = 0
        while i < len(self.array)-1:
            weights.append(rowStochMatrix(np.random.random((self.array[i+1], self.array[i]))))
            biases.append(normalizeArray(np.random.random((self.array[i+1],1))))
            tempZerosArray = np.zeros(self.array[i+1], 1)
            tempZerosMatrix = np.zeros(self.array[i+1], self.array[i])
            m1Biases.append(tempZerosArray)
            m2Biases.append(tempZerosArray)
            m1Weights.append(tempZerosMatrix)
            m2Weights.append(tempZerosMatrix)
            i = i + 1
        self.weights = weights
        self.biases = biases
        self.m1B = m1Biases
        self.m2B = m2Biases
        self.m1W = m1Weights
        self.m2W = m2Weights
        return self.weights, self.biases, self.m1B, self.m2B, self.m1W, self.m2W

    #Feedback loop
    def feedBack(self, label, outputArray, timeStep):
        deltaLayers = [] #Start with the the delta values of each layer as an empty set
        outputA = vc.Vector(outputArray) #Create a vector for the output array
        labelA = vc.Vector(createLabelArray(label)) #Create a vector for the label (which entry is a 1, others 0)
        outputA.vecDiff(labelA.array) #Get the difference between the desired value (label) and what we got from NN
        tempRet = vc.Vector(self.zeds[len(self.zeds) - 1].array) #Pull out the last layers zed values
        tempRet.sigmoidDerArray() #Find the derivative of the sigmoid array at those points
        tempRet.matrixProduct(np.diag(outputA.array)) #Find the delta values of the last layer
        deltaLayers.append(tempRet) #Store the vector of the delta values
        i = 0
        while i < self.layersNet - 1:
            tempRet = vc.Vector(self.zeds[len(self.zeds) - 2 - i].array) #Keep moving backwards and get the rest deltas 
            tempRet.sigmoidDerArray()
            prevDelta = vc.Vector(deltaLayers[i].array)
            prevDelta.matrixProduct(np.transpose(self.weights[self.layersNet - 2 - i]))
            tempRet.matrixProduct(np.diag(prevDelta.array))
            deltaLayers.append(tempRet)
            i = i + 1
        weightDiff = []
        i = 0 
        while i < self.layersNet - 1:
            zedCall = vc.Vector(self.zeds[i + 1].array)
            delCall = vc.Vector(deltaLayers[self.layersNet - i - 1].array)
            weightDiff.append(rowStochMatrix(delCall.matrixMult(zedCall.sigmoidArray())))
            i = i + 1
        i = 0
        while i < self.layersNet - 1:
            #self.weights[i] = rowStochMatrix(matrixDiff(self.weights[i], np.transpose(weightDiff[i])))
            #tempB = vc.Vector(self.biases[self.layersNet - i - 2])
            self.sGradDescentArray(deltaLayers[i].array, timeStep, i) #Apply SGD algorithm for arrays
            self.sGradDescentMatrix(weightDiff[i], timeStep, i) #Apply SGD algorithm for matrices
            #tempB.vecDiff(deltaLayers[i].array)
            #tempB.normalizeVector()
            #self.biases[self.layersNet - i - 2] = tempB.array
            i = i + 1
        return 1 

    #ADAM stochastic gradient descent algorithm for an array (see https://arxiv.org/pdf/1412.6980.pdf)
    def sGradDescentArray(self, gradArray, tStep, layer):
            tStep = tStep + 1
            alpha = self.learningRates[0]
            beta1 = self.learningRates[1]
            beta2 = self.learningRates[2]
            epsilon = self.learningRates[3]
            tempVector = vc.Vector(gradArray)
            #BE CAREFUL, CHECK WHICH LAYER WE NEED FOR m1B, etc...
            self.m1B[layer] = beta1*self.m1B[layer] + (1-beta1)*gradArray
            self.m2B[layer] = beta2*self.m2B[layer] + (1-beta2)*((tempVector.vecSq()).array)
            mhat1B = vcVector(self.m1B[layer])
            mhat1B.multiConstant((1-beta1**tStep)**(-1))
            mhat2B = vc.Vector(self.m2B[layer])
            mhat2B.multiConstant((1-beta2**tStep)**(-1))
            tempVector = vc.Vector(self.biases[self.layersNet-layer-2] - alpha*mhat1B.vecDivision(
                    (mhat2B.vecSqrt()).addConstant(epsilon)))
            tempVector.normalizeVector()
            self.biases[self.layersNet-layer-2] = tempVector.array
    

    #ADAM stochastic gradient descent algorithm for a matrix
    def sGradDescentMatrix(self, gradMatrix, tStep, layer):
            tStep = tStep + 1
            alpha = self.learningRates[0]
            beta1 = self.learningRates[1]
            beta2 = self.learningRates[2]
            epsilon = self.learningRates[3]
            tempVector = vc.Vector(gradArray)
            self.m1W[layer] = beta1*self.m1W[layer] + (1-beta1)*gradArray
            self.m2W[layer] = beta2*self.m2W[layer] + (1-beta2)*((tempVector.vecSq()).array)
            mhat1W = vcVector(self.m1W[layer])
            mhat1W.multiConstant((1-beta1**tStep)**(-1))
            mhat2W = vc.Vector(self.m2B[layer])
            mhat2W.multiConstant((1-beta2**tStep)**(-1))
            tempVector = vc.Vector(self.biases[self.layersNet-layer-2] - alpha*mhat1W.vecDivision(
                    (mhat2W.vecSqrt()).addConstant(epsilon)))
            tempVector.normalizeVector()
            self.biases[self.layersNet-layer-2] = tempVector.array


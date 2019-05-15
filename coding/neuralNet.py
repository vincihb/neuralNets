import numpy as np
import math 
import mnistLoad
import vector as vc
import matrix as mx

class net(object):

    zeds = []
    layersNet = 0

    #Initialise the neural network
    def __init__(self, array):
        self.array = array
    
    #Aside from the other initialisation function, we have a seperate function for initialising the weights
    #biases, and the moment estimates for the stochastic gradient descent algorithm
    def initialiseNet(self, learningRates):
        print "Initialising the neural net"
        self.alpha = learningRates[0]
        self.beta1 = learningRates[1]
        self.beta2 = learningRates[2]
        self.epsilon = learningRates[3]
        weights = []
        biases = []
        m1Biases = []
        m2Biases = []
        m1Weights = []
        m2Weights = []
        self.layersNet = len(self.array)
        i = 0
        while i < len(self.array) - 1:
            wTemp = mx.Matrix(np.random.random((self.array[i+1], self.array[i])))
            weights.append(wTemp.rowStochMatrix())
            biases.append(normalizeArray(np.random.random(self.array[i+1])))
            tempZerosArray = np.zeros(self.array[i+1])
            tempZerosMatrix = np.zeros((self.array[i+1], self.array[i]))
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

   
    #Nifty little function to describe what the attributes of the neural network are
    def description(self):
        print(self.array)
        print(self.zeds)
        print(self.layersNet)
    
    #Feedforward loop
    def feedForward(self, inputArray, flag):
        #print self.weights[0][0][0] #For debugging, printing out what the weight is 
        activations = vc.Vector(inputArray) #Create a vector for the activations 
        tempRet = vc.Vector(inputArray) #Create a temporary vector for the input array
        if flag == 0: #Check if this is the first training loop
            self.zeds.append(activations.array) #Add the activations of the first layer as zeds, need 
            #later for weightDiff array
            i = 0
            while i < self.layersNet - 1:
                tempRet.matrixProduct(self.weights[i]) #Multiply the weights with the current activations
                tempRet.vecAdd(self.biases[i]) #Add the biases
                tempRet.normalizeVector() #Normalize
                self.zeds.append(tempRet.array) #And we get the zeds of layer i ((i+1)th layer)
                activations = tempRet.sigmoidArray() #Activations are just the sigmoid of the zeds we created
                i = i + 1 #Iterate to go to the next layer
        else:
            self.zeds[0] = inputArray
            i = 0
            while i < self.layersNet - 1:
                tempRet.matrixProduct(self.weights[i])
                tempRet.vecAdd(self.biases[i])
                tempRet.normalizeVector()
                self.zeds[i+1] = tempRet.array
                activations = tempRet.sigmoidArray()
                #self.zeds[i+1] = (normalizeArray(arrayAdd(self.weights[i].dot(activations),
                #    self.biases[i])))
                #activations = normalizeArray(sigmoidArray(self.zeds[i+1]))
                i = i + 1
        return activations

    #Feedback loop
    def feedBack(self, label, outputArray, timeStep):
        deltaLayers = [] #Start with the the delta values of each layer as an empty set
        outputA = vc.Vector(outputArray) #Create a vector for the output array
        labelA = vc.Vector(createLabelArray(label)) #Create a vector for the label (which entry is a 1, others 0)
        outputA.vecDiff(labelA.array) #Get the difference between the desired value (label) and what we got from NN
        tempRet = vc.Vector(self.zeds[len(self.zeds)-1]) #Pull out the last layers zed values
        tempRet.sigmoidDerArray() #Find the derivative of the sigmoid array at those points
        tempRet.matrixProduct(np.diag(outputA.array)) #Find the delta values of the last layer
        deltaLayers.append(tempRet) #Store the vector of the delta values
        i = 0
        while i < len(self.zeds) - 2: #self.zeds[0] is the inputArray, so we have to take away 2
            tempRet = vc.Vector(self.zeds[len(self.zeds)-2-i])
            tempRet.sigmoidDerArray()
            prevDelta = vc.Vector(deltaLayers[i].array)
            prevDelta.matrixProduct(np.transpose(self.weights[self.layersNet-2-i]))
            prevDelta.matrixProduct(np.diag(tempRet.array))
            deltaLayers.append(prevDelta)
            i = i + 1
        weightDiff = []
        i = 0 
        while i < self.layersNet - 1:
            zedCall = vc.Vector(self.zeds[i])
            delCall = vc.Vector(deltaLayers[self.layersNet-i-2].array)
            wDTemp = mx.Matrix(delCall.matrixMult(zedCall.sigmoidArray()))
            weightDiff.append(wDTemp.rowStochMatrix())
            i = i + 1
            #zedCall = vc.Vector(self.zeds[i+1])
            #delCall = vc.Vector(deltaLayers[self.layersNet-i-1].array)
            #weightDiff.append(rowStochMatrix(delCall.matrixMult(zedCall.sigmoidArray())))
            #i = i + 1
        i = 0
        while i < self.layersNet - 1:
            print "..... In feedback loop at layer " + str(i)
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
            tempVector = vc.Vector(gradArray)
            self.m1B[self.layersNet-layer-2] = self.beta1*np.asarray(self.m1B[self.layersNet-layer-2]
                    ) + (1-self.beta1)*np.asarray(gradArray)
            self.m2B[self.layersNet-layer-2] = self.beta2*np.asarray(self.m2B[self.layersNet-layer-2]
                    ) + (1-self.beta2)*np.asarray(tempVector.vecSq())
            mhat1B = vc.Vector(self.m1B[self.layersNet-layer-2])
            mhat1B.multiConstant((1-self.beta1**tStep)**(-1))
            mhat2B = vc.Vector(self.m2B[self.layersNet-layer-2])
            mhat2B.multiConstant((1-self.beta2**tStep)**(-1))
            mhat2B.vecSqrt()
            mhat2B.addConstant(self.epsilon)
            #self.alpha = self.alpha*math.sqrt(1 - self.beta2**tStep)/(1-self.beta1**tStep) 
            tempVector = vc.Vector(self.biases[self.layersNet-layer-2] 
                    - self.alpha*np.asarray(mhat1B.vecDivision(mhat2B))) 
            tempVector.normalizeVector()
            self.biases[self.layersNet-layer-2] = tempVector.array
    

    #ADAM stochastic gradient descent algorithm for a matrix
    def sGradDescentMatrix(self, gradMatrix, tStep, layer):
            tStep = tStep + 1
            gMatrix = mx.Matrix(gradMatrix)
            m1W = mx.Matrix(self.m1W[layer])
            m1W.matrixMConstant(-self.beta1)
            gMatrix.matrixMConstant(1 - self.beta1)
            self.m1W[layer] = gMatrix.matrixDiff(m1W)
            #self.m1W[layer] = self.beta1*self.m1W[layer] + (1-self.beta1)*gradMatrix
            #self.m2W[layer] = self.beta2*self.m2W[layer] + (1-self.beta2)*(matrixSq(gradMatrix))
            gMatrix = mx.Matrix(gradMatrix)
            gMatrix.matrixSq()
            m2W = mx.Matrix(self.m2W[layer])
            m2W.matrixMConstant(-self.beta2)
            gMatrix.matrixMConstant(1 - self.beta2)
            self.m2W[layer] = gMatrix.matrixDiff(m2W)
            mhat1W = mx.Matrix(self.m1W[layer])
            mhat1W.matrixMConstant((1-self.beta1**tStep)**(-1))
            mhat2W = mx.Matrix(self.m2W[layer])
            mhat2W.matrixMConstant((1-self.beta2**tStep)**(-1))
            #self.alpha = self.alpha*math.sqrt(1 - self.beta2**tStep)/(1-self.beta1**tStep)
            #mhat1W = matrixMConstant(self.m1W[layer], (1-self.beta1**tStep)**(-1))
            #mhat2W = matrixMConstant(self.m2W[layer], (1-self.beta2**tStep)**(-1))
            mhat2W.matrixSqrt()
            mhat2W.matrixMConstant(self.epsilon)
            mhat1W.matrixDivide(mhat2W)
            mhat1W.matrixMConstant(self.alpha)
            tempMatrix = mx.Matrix(self.weights[layer])
            tempMatrix.matrixDiff(mhat1W)
            #tempMatrix = self.weights[layer] - matrixMConstant(matrixDivide(mhat1W, 
            #        matrixMConstant(matrixSqrt(mhat2W), self.epsilon)), self.alpha)
            tempMatrix.rowStochMatrix()
            self.weights[layer] = tempMatrix.matrix

#misc functions
def createLabelArray(label):
    array = []
    i = 0
    while i < 10:
        array.append(0)
        i = i + 1
    array[label] = 1
    return array

def matrixDivide(matrix1, matrix2):
    matrixRet = []
    i = 0
    while i < len(matrix1):
        j = 0 
        arrayR = []
        while j < len(matrix1[0]):
            arrayR.append(matrix1[i][j]/matrix2[i][j])
            j = j + 1 
        matrixRet.append(arrayR)
        i = i + 1 
    return matrixRet

def matrixSqrt(matrix1):
    matrixRet = []
    i = 0
    while i < len(matrix1):
        j = 0
        arrayR = []
        while j < len(matrix1[0]):
            arrayR.append(matrix1[i][j]**(0.5))
            j = j + 1
        matrixRet.append(arrayR)
        i = i + 1
    return matrixRet

def matrixMConstant(matrix1, c):
    matrixRet = []
    i = 0
    while i < len(matrix1):
        j = 0
        arrayR = []
        while j < len(matrix1[0]):
            arrayR.append(matrix1[i][j]*c)
            j = j + 1
        matrixRet.append(arrayR)
        i = i + 1
    return matrixRet

def matrixSq(matrix1):
    matrixRet = []
    i = 0
    while i < len(matrix1):
        j = 0
        arrayR = []
        while j < len(matrix1[0]):
            arrayR.append(matrix1[i][j]*matrix1[i][j])
            j = j + 1
        matrixRet.append(arrayR)
        i = i + 1
    return matrixRet

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

def rowStochMatrix(arrayM):
    i = 0
    while i < len(arrayM):
        arrayM[i] = normalizeArray(arrayM[i])
        i = i + 1
    return arrayM

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

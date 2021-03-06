import math
from numba import vectorize

class Vector(object):

    def __init__(self, array):
        self.array = array

    #Standard dot product between two vectors
    def dotProduct(self, array1):
        dotPro = 0
        i = 0
        while i < len(self.array):
            dotPro = dotPro + (self.array[i] * array1[i])
            i = i + 1
        return dotPro
    
    #Multiply a matrix with a vector
    def matrixProduct(self, matrix1):
        arrayRet = []
        i = 0
        while i < len(matrix1):
            arrayRet.append(self.dotProduct(matrix1[i]))
            i = i + 1
        self.array = arrayRet
        return arrayRet
    
    #Create a matrix with two vectors
    def matrixMult(self, array2):
        array = []
        i = 0
        while i < len(self.array):
            j = 0
            array.append([])
            while j < len(array2):
                array[i].append(self.array[i]*array2[j])
                j = j + 1
            i = i + 1
        return array
    
    #Add two vectors
    def vecAdd(self, array2):
        arrayRet = []
        i = 0
        while i < len(self.array):
            arrayRet.append(self.array[i] + array2[i])
            i = i + 1
        self.array = arrayRet
        return arrayRet
    
    #Subtract self with another vector 
    def vecDiff(self, array2):
        arrayRet = []
        i = 0
        while i < len(self.array):
            arrayRet.append(self.array[i] - array2[i])
            i = i + 1
        self.array = arrayRet
        return arrayRet
    
    #Square a vector
    def vecSq(self):
        arrayRet = []
        i = 0
        while i < len(self.array):
            arrayRet.append(self.array[i]*self.array[i])
            i = i + 1
        self.array = arrayRet
        return arrayRet
    
    #Square root of a vector
    def vecSqrt(self):
        arrayRet = []
        i = 0
        while i < len(self.array):
            arrayRet.append(self.array[i]**(0.5))
            i = i + 1
        self.array = arrayRet
        return arrayRet
    
    #Divide two vectors
    def vecDivision(self, vector2):
        arrayRet = []
        i = 0
        while i < len(self.array):
            arrayRet.append(self.array[i]/(vector2.array[i]))
            i = i + 1
        self.array = arrayRet
        return arrayRet
    
    #Add a constant to each entry
    def addConstant(self, c):
        arrayRet = []
        i = 0
        while i < len(self.array):
            arrayRet.append(self.array[i] + c)
            i = i + 1
        self.array = arrayRet
        return arrayRet

    #Multiply by a constant 
    def multiConstant(self, c):
        arrayRet = []
        i = 0
        while i < len(self.array):
            arrayRet.append(self.array[i]*c)
            i = i + 1
        self.array = arrayRet
        return arrayRet

    #Normalize vector
    def normalizeVector(self):
        i = 0
        sumArray = 0
        arrayRet = []
        while i < len(self.array):
            sumArray = self.array[i] + sumArray
            i = i + 1
        i = 0
        while i < len(self.array):
            #self.array[i] = (self.array[i]/sumArray)
            arrayRet.append(self.array[i]/sumArray)
            i = i + 1
        self.array = arrayRet
        return arrayRet

    #Applies sigmoid derivative on each element
    def sigmoidDerArray(self):
        arrayRet = []
        i = 0
        while i < len(self.array):
            arrayRet.append(sigmoidDerivative(self.array[i]))
            i = i + 1
        arrayRet = normalizeArray(arrayRet)
        self.array = arrayRet
        return arrayRet
    
    #Applies sigmoid on each element
    def sigmoidArray(self):
        arrayRet = []
        i = 0
        while i < len(self.array):
            arrayRet.append(sigmoid(self.array[i]))
            i = i + 1
        arrayRet = normalizeArray(arrayRet)
        self.array = arrayRet
        return arrayRet

#misc functions
def sigmoid(x):
    return 1.0/(1+math.exp(-x))

def sigmoidDerivative(x):
    return math.exp(-x)/((1+math.exp(-x))**2)

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


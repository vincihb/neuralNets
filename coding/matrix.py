import numpy as np
import math
import vector as vc
import multiprocessing as mp


#def uwmatrixSqrt(cl):
#    pool = mp.Pool(mp.cpu_count())
#    result = pool.apply(matrixSqrt())
#    pool.close()
#    return result

class Matrix(object):
    
    def __init__(self, matrix1):
        self.matrix = matrix1
    
    def matrixDivide(self, matrix2):
        matrixRet = []
        i = 0
        while i < len(self.matrix):
            j = 0 
            arrayR = []
            while j < len(self.matrix[0]):
                arrayR.append(self.matrix[i][j]/matrix2.matrix[i][j])
                j = j + 1 
            matrixRet.append(arrayR)
            i = i + 1 
        self.matrix = matrixRet
        return matrixRet

    def matrixSqrt(self):
        matrixRet = []
        i = 0
        while i < len(self.matrix):
            j = 0
            arrayR = []
            while j < len(self.matrix[0]):
                arrayR.append(self.matrix[i][j]**(0.5))
                j = j + 1
            matrixRet.append(arrayR)
            i = i + 1
        self.matrix = matrixRet
        return matrixRet

    def matrixMConstant(self, c):
        matrixRet = []
        i = 0
        while i < len(self.matrix):
            j = 0
            arrayR = []
            while j < len(self.matrix[0]):
                arrayR.append(self.matrix[i][j]*c)
                j = j + 1
            matrixRet.append(arrayR)
            i = i + 1
        self.matrix = matrixRet
        return matrixRet

    def matrixSq(self):
        matrixRet = []
        i = 0
        while i < len(self.matrix):
            j = 0
            arrayR = []
            while j < len(self.matrix[0]):
                arrayR.append(self.matrix[i][j]*self.matrix[i][j])
                j = j + 1
            matrixRet.append(arrayR)
            i = i + 1
        self.matrix = matrixRet
        return matrixRet

    
    def matrixDiff(self, matrix2):
        matrixRet = []
        i = 0
        while i < len(self.matrix):
            j = 0
            arrayR = []
            while j < len(self.matrix[0]):
                arrayR.append(abs(self.matrix[i][j] - matrix2.matrix[i][j]))
                j = j + 1
            matrixRet.append(arrayR)
            i = i + 1
        self.matrix = matrixRet
        return matrixRet

    
    def rowStochMatrix(self):
        matrixRet = []
        i = 0
        while i < len(self.matrix):
            matrixRet.append(normalizeArray(self.matrix[i]))
            i = i + 1
        self.matrix = matrixRet
        return matrixRet

#Misc. functions

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

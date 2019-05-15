import vector as vc
import numpy as np

arrayV = []
v = vc.Vector([1., 2., 3.])


m = [[2, 0, 0],
    [0, 2, 0]]

arrayV.append(v.array)

print v.matrixProduct(m)
print v.array
print v.normalizeVector()

arrayV.append(v.array)

tempArray = [2., 2., 1.]
v = vc.Vector(tempArray)
v.vecSq()
print tempArray
print v.array
v.vecSqrt()
print v.array
v.addConstant(5)
print v.array
v.multiConstant(10**-1)
print v.array
print arrayV[0]
print arrayV[1]

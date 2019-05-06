import vector as vc
import numpy as np

arrayV = []
v = vc.Vector([])
v.array = [1.,2.,3.]

arrayV.append(v)

m = [[2, 0, 0],
    [0, 2, 0]]

print v.matrixProduct(m)
print v.array
print v.normalizeVector()

v.array = [1., 2., 3.]
tempArray = [2., 2., 1.]
v2 = vc.Vector(tempArray)
v2.vecSq()
print tempArray
print v2.array
v2.vecSqrt()
print v2.array
v2.addConstant(5)
print v2.array
v2.multiConstant(10**-1)
print v2.array


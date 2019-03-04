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

v = vc.Vector([2., 2., 1.])

print v.array
print arrayV[0].array

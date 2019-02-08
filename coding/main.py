import neuralNet
import numpy as np
import mnistLoad

n1 = neuralNet.net([784, 200, 50, 10])
n1.description()

w, b = n1.initialiseNet(n1.array)

imagesTrain, labelsTrain, imagesTest, labelsTest = mnistLoad.initialiseMnist()

print(n1.feedForward(imagesTrain[0], w, b, n1.array))


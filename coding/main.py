import neuralNet
import numpy as np
import mnistLoad

n1 = neuralNet.net([784, 200, 50, 10])

w, b = n1.initialiseNet(n1.array)

imagesTrain, labelsTrain, imagesTest, labelsTest = mnistLoad.initialiseMnist()

activations = n1.feedForward(imagesTrain[0])

print activations

print(n1.feedBack(labelsTrain[0], activations))

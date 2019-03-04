import neuralNet
import numpy as np
import mnistLoad
import vector as vc

n1 = neuralNet.net([784, 200, 50, 10])

w, b = n1.initialiseNet()

imagesTrain, labelsTrain, imagesTest, labelsTest = mnistLoad.initialiseMnist()

i = 0
while i < len(imagesTrain):
    print "On level " + str(i)
    activations = n1.feedForward(imagesTrain[i], i)
    print labelsTrain[i]
    print activations
    n1.feedBack(labelsTrain[i], activations)
    i = i + 1

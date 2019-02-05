import neuralNet
import numpy as np
import mnistLoad

n1 = neuralNet.net()
n1.arrayOfNeurons = [784, 200, 50, 10]
n1.description()

w, b = neuralNet.initialiseNet(n1.arrayOfNeurons)

imagesTrain, labelsTrain, imagesTest, labelsTest = mnistLoad.initialiseMnist()

print(neuralNet.feedForward(imagesTrain[0], w, b, n1.arrayOfNeurons))


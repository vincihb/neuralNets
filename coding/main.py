import neuralNet
import numpy as np
import mnistLoad
import vector as vc

#Recommended learning rate constants, please refer to ADAM paper (https://arxiv.org/pdf/1412.6980.pdf)
alpha = 0.001
beta1 = 0.9
beta2 = 0.999
epsilon = 10**(-8)
learningRates = [alpha, beta1, beta2, epsilon]

n1 = neuralNet.net([784, 200, 50, 10])

n1.initialiseNet(learningRates)

imagesTrain, labelsTrain, imagesTest, labelsTest = mnistLoad.initialiseMnist()


i = 0
while i < len(imagesTrain):
    print "On level " + str(i)
    activations = n1.feedForward(imagesTrain[i], i)
    print "Expected output: " + str(labelsTrain[i])
    print activations
    n1.feedBack(labelsTrain[i], activations, i)
    i = i + 1

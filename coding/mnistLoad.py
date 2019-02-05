from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def initialiseMnist():
    mndata = MNIST('mnist')

    imagesTrain, labelsTrain = mndata.load_training()

    imagesTest, labelsTest = mndata.load_testing()

    return imagesTrain, labelsTrain, imagesTest, labelsTest
# or
#images, labels = mndata.load_testing()

#newImage = np.reshape(images[0], (-1, 28))

#plt.imshow(newImage, cmap="gray")

#plt.show()



#plot.imshow((28, 28), images[0])

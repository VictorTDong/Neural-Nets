import network
import idx2numpy
import numpy as np


# testing data
testingImageFile = "data/t10k-images.idx3-ubyte"
testingLabelFile = "data/t10k-labels.idx1-ubyte"

# returns the number of entries in the file, as well as a list of integers
# representing the correct label for each entry
def getLabels(labelfile):
    file = open(labelfile, 'rb')
    file.read(4)
    n = int.from_bytes(file.read(4), byteorder='big') # number of entries
    
    labelarray = bytearray(file.read())
    labelarray = [b for b in labelarray]    # convert to ints
    file.close()
    
    return n, labelarray

# returns a list containing the pixels for each image, stored as a (784, 1) numpy array
def getImgData(imagefile):
    # returns an array whose entries are each (28x28) pixel arrays with values from 0 to 255.0
    # standarizes the values
    images = idx2numpy.convert_from_file(imagefile) 
    images = images/255
    # We want to flatten each image from a 28 x 28 to a 784 x 1 numpy array
    features = [np.reshape(singleImage,(784,1)) for singleImage in images[0:]]

    return features

def prepData():
    # reads data
    ntest, test_labels = getLabels(testingLabelFile)
    testingFeatures = getImgData(testingImageFile)

    testingLabels = test_labels[:ntest]

    # zips array

    testingData = zip(testingFeatures,testingLabels)

    return (testingData)


testingData = prepData()

input = network.loadFromFile("part1.pkl")
wrongIndex, numberOfWrongs = (input.findImage(testingData))
print("This is the first three indexs wrong: {} out of {} wrong indexes".format(wrongIndex[:3], numberOfWrongs))



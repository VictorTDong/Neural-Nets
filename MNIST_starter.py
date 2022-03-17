import numpy as np
import time
import idx2numpy
import network

# converts a 1d python list into a (1,n) row vector
def rv(vec):
    return np.array([vec])
    
# converts a 1d python list into a (n,1) column vector
def cv(vec):
    return rv(vec).T
        
# creates a (size,1) array of zeros, whose ith entry is equal to 1    
def onehot(i, size):
    vec = np.zeros(size)
    vec[i] = 1
    return cv(vec)


##################################################
# NOTE: make sure these paths are correct for your directory structure

# training data
trainingImageFile = "data/train-images.idx3-ubyte"
trainingLabelFile = "data/train-labels.idx1-ubyte"

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


# reads the data from the four MNIST files,
# divides the data into training and testing sets, and encodes the training vectors in onehot form
# returns a tuple (trainingData, testingData), each of which is a zipped array of features and labels
def prepData():
    # reads data
    ntrain, train_labels = getLabels(trainingLabelFile)
    ntest, test_labels = getLabels(testingLabelFile)
    trainingFeatures = getImgData(trainingImageFile)
    testingFeatures = getImgData(testingImageFile)

    trainingLabels = [onehot(label, 10) for label in train_labels[:ntrain]]
    testingLabels = test_labels[:ntest]

    # zips array
    trainingData = zip(trainingFeatures,trainingLabels)
    testingData = zip(testingFeatures,testingLabels)

    return (trainingData, testingData)
    

###################################################

trainingData, testingData = prepData()

net = network.Network([784,35,10])
start = time.time()
net.SGD(trainingData, 50, 10, 5.6, test_data = testingData)
end = time.time()

print("The time of execution of above program is :", end-start)
network.saveToFile(net, "part1.pkl")



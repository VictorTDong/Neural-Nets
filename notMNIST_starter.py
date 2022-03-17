import numpy as np
import network
import time


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

    
#################################################################

# reads the data from the notMNIST.npz file,
# divides the data into training and testing sets, and encodes the training vectors in onehot form
# returns a tuple (trainingData, testingData), each of which is a zipped array of features and labels
def prepData():
    # loads the four arrays specified.
    # train_features and test_features are arrays of (28x28) pixel values from 0 to 255.0
    # train_labels and test_labels are integers from 0 to 9 inclusive, representing the letters A-J
    with np.load("data/notMNIST.npz", allow_pickle=True) as f:
        train_features, train_labels = f['x_train'], f['y_train']
        test_features, test_labels = f['x_test'], f['y_test']
        
    train_features, test_features = train_features/255.0, test_features/255.0

    trainFeaturesFlattened = [np.reshape(trainImage,(784,1)) for trainImage in train_features[0:]]
    testFeaturesFlattened = [np.reshape(testImage,(784,1)) for testImage in test_features[0:]]

    trainingLabels = [onehot(label, 10) for label in train_labels[0:]]

    trainingData = zip(trainFeaturesFlattened,trainingLabels)
    testingData = zip(testFeaturesFlattened,test_labels)

    return (trainingData, testingData)
    
###################################################################


trainingData, testingData = prepData()

net = network.Network([784,30,10])
start = time.time()
net.SGD(trainingData, 50, 10, 5, test_data = testingData)
end = time.time()

print("The time of execution of above program is :", end-start)

network.saveToFile(net, "part2.pkl")
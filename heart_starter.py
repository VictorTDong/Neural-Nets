import csv
from pickle import FRAME
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

# given a data point, mean, and standard deviation, returns the z-score
def standardize(x, mu, sigma):
    return ((x - mu)/sigma)

def openFile(filename):
    with open(filename, newline='') as datafile:
        reader = csv.reader(datafile)        
        next(reader, None)  # skip the header row
        sampleArray = []
        for row in reader:
            sampleArray.append(row)
    return sampleArray

def getMeanAndStdDev(allSamples, i):
    n = 0
    totalSum = []
    for row in allSamples:
        totalSum.append(float(row[i]))
        n = n + 1
    mean = (sum(totalSum))/n
    stdDev = np.std(totalSum, axis=0)
    return mean, stdDev
##############################################

# given a list with the features and label for a sample (row of the csv),
# converts it to a numeric feature vector and an integer label
# returns the tuple (feature, label)
def getDataFromSample(allSamples, sample):

    # sdp
    sdpMean, sdpStd = getMeanAndStdDev(allSamples, 1)
    sdp = cv([standardize(float(sample[1]), sdpMean, sdpStd)])

    # tobacco
    tabaccoMean, tabaccoStd = getMeanAndStdDev(allSamples, 2)
    tobacco = cv([standardize(float(sample[2]), tabaccoMean, tabaccoStd)])

    # ldl
    ldlMean, ldlStd = getMeanAndStdDev(allSamples, 3)
    ldl = cv([standardize(float(sample[3]), ldlMean, ldlStd)])

    # adiposity
    adiposityMean, adiposityStd = getMeanAndStdDev(allSamples, 4)
    adiposity = cv([standardize(float(sample[4]), adiposityMean, adiposityStd)])

    # famhist
    if (sample[5] == "Present"):
        famhist = cv([1])
    elif (sample[5] == "Absent"):
        famhist = cv([0])
    else:
        print("Data processing error. Exiting")
        quit()

    # typea
    typeaMean, typeaStd = getMeanAndStdDev(allSamples, 6)
    typea = cv([standardize(float(sample[6]), typeaMean, typeaStd)])
    # obesity
    obesityMean, obesityStd = getMeanAndStdDev(allSamples, 7)
    obesity = cv([standardize(float(sample[7]), obesityMean, obesityStd)])
    # alcohol
    sdpMean, sdpStd = getMeanAndStdDev(allSamples, 8)
    alcohol = cv([standardize(float(sample[8]), sdpMean, sdpStd)])

    # age
    normalizedAge = float(sample[9])/64
    ageMean, ageStd = getMeanAndStdDev(allSamples, 9)
    age = cv([standardize(normalizedAge, ageMean, ageStd)])

    features = np.concatenate((sdp,tobacco,ldl,adiposity,famhist,typea,obesity,alcohol,age), axis = 0)
    label = int(sample[10])

    return (features, label)

##############################################

# reads number of data points, feature vectors and their labels from the given file
# and returns them as a tuple
def readData(filename):
    with open(filename, newline='') as datafile:
        reader = csv.reader(datafile)        
        next(reader, None)  # skip the header row

        n = 0
        features = []
        labels = []
        allSamples = openFile(filename)
        for row in reader:
            featureVec, label = getDataFromSample(allSamples, row)
            features.append(featureVec)
            labels.append(label)
            n = n + 1
    
    return n, features, labels


################################################

# reads the data from the heart.csv file,
# divides the data into training and testing sets, and encodes the training vectors in onehot form
# returns a tuple (trainingData, testingData), each of which is a zipped array of features and labels
def prepData():

    n, features, labels = readData('data/heart.csv')

    ntrain = int(n * 5/6)
    ntest = n - ntrain

    trainingFeatures = features[:ntrain]
    trainingLabels = [onehot(label,2) for label in labels[:ntrain]]

    print(f"Number of training samples: {ntrain}")
    
    testingFeatures = features[ntrain:]
    testingLabels = labels[ntrain:]

    print(f"Number of testing samples: {ntest}")

    trainingData = zip(trainingFeatures,trainingLabels)
    testingData = zip(testingFeatures,testingLabels)

    return (trainingData, testingData)


###################################################


trainingData, testingData = prepData()

net = network.Network([9,10,2])
start = time.time()
net.SGD(trainingData, 100, 10, .27, test_data = testingData)
end = time.time()

print("The time of execution of above program is :", end-start)

network.saveToFile(net, "part3.pkl")

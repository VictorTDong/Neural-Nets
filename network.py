import numpy as np
import random
import pickle

'''
Author: Janet Leahy

Network class taken from "Neural Networks and Deep Learning"
Michael Nielsen, 2015

Modifications made to allow saving and loading trained networks to file
'''


# applies the sigmoid function to each entry in the given array
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

# applies the derivative of the sigmoid function to each entry in the given array
def sigmoid_prime(z):
    return sigmoid(z)*(1 - sigmoid(z))


##################################################


# saves the network to the given filename
def saveToFile(net, filename):
    data = [net.sizes, net.biases, net.weights]
    with open(filename, 'wb') as outfile:
        pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)


# loads a network from the data in the given filename
def loadFromFile(filename):
    with open(filename, 'rb') as infile:
        data = pickle.load(infile)

    return Network(data[0], data[1], data[2])

################################################

# Code imported from https://github.com/MichalDanielDobrzanski/DeepLearningPython/blob/master/network.py
# based on the Nielsen textbook, Chapter 1
# modifications made to allow saving trained nets to a file


class Network(object):

    def __init__(self, sizes, biases = None, weights = None):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = biases
        self.weights = weights
        
        if (biases == None):
            # initialize randomly from a normal distribution
            self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
            
        if (weights == None):
            # initialize randomly from a normal distribution
            self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""

        training_data = list(training_data)
        n = len(training_data)
        print(f"Length of training data: {n}")

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
            print(f"Length of test data: {n_test}")
            print("Initial performance : {} / {}".format(self.evaluate(test_data),n_test))

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test))
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

# This will return the number of tests inputs in which the neural network outputs the incorrect result.
# This will also return the array containing a all the indexs which contain a wrong prediction
    def findImage(self, test_data):
        wrongLabels = []
        index = 0
        numberOfWrongs = 0
        if test_data:
            test_data = list(test_data)
            results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
            #print(results)
            for (x, y) in results:
                if x != y:
                    wrongLabels.append(["Index: ", index, "Guess :", x, "Correct :", y])
                    numberOfWrongs += 1
                index += 1
        return wrongLabels, numberOfWrongs

 ################################################


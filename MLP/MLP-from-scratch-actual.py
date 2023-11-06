import seaborn as sns
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as pd
import matplotlib.cm as cm
from sklearn.datasets import load_iris
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

sns.set(style='whitegrid'); sns.set_context('talk')

# Load the iris dataset
iris = load_iris()

samples, features = iris.data.shape

train_percent = 80
test_percent = 20

dataset = np.column_stack((iris.data, iris.target))
dataset = list(dataset)
random.shuffle(dataset)

def separate_data():
    1 = dataset[0:40]
    t1 = dataset[40:50]
    2 = dataset[50:90]
    t2 = dataset[90:100]
    3 = dataset[100:140]
    t3 = dataset[140:150]
    train = np.concatenate((1, 2, 3))
    test = np.concatenate((t1, t2, t3))
    return train, test

train_files, test_files = separate_data()

class MLP(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.inputLayer = 4 # We have 4 different types of data to feed as inputs
        self.hiddenLayer = 5
        self.outputLayer = 3 # There are 3 clasfications for this dataset
        self.learningRate = 0.01
        self.maxEpochs = 1000
        self.biasHiddenValue = -1
        self.biasOutputValue = -1
        self.activation = self.activateFunction
        self.derivative = self.derivativeFunction

        # Starting weights and biases
        self.weightHidden = self.starting_weights(self.hiddenLayer, self.inputLayer)
        self.weightOutput = self.starting_weights(self.OutputLayer, self.hiddenLayer)
        self.biasHidden = np.array([self.BiasHiddenValue for i in range(self.hiddenLayer)])
        self.biasOutput = np.array([self.BiasOutputValue for i in range(self.OutputLayer)])
        self.numClasses = 3 

    pass

    def starting_weights(self, n, m):
        return np.random.uniform(-1, 1, (n, m))
    
    # Sigmoid activation function
    activationFunction = (lambda x: 1/(1 + np.exp(-x)))

    # Derivative of the sigmoid function
    derivativeFunction = (lambda x: x * (1 - x))

    def Backpropagation_Algorithm(self, x):
        # Error for output layer
        output = []
        errOutput = self.output - self.outputL2
        output = ((-1)*errOutput) * self.derivative(self.outputL2)

        # Updating weights in output layer and hidden layer
        arr = []
        for i in range(self.hiddenLayer):
            for j in range(self.outputLayer):
                self.weightOutput[i][j] -= (self.learningRate * (output[j] * self.outputL1[i]))
                self.biasOutput[j] -= (self.learningRate * output[j])

        # Error for hidden layer
        hidden = np.matmul(self.weightOutput, output) * self.derivative(self.outputL1)

        # Updating weights in hidden layer and input layer
        for i in range(self.inputLayer):
            for j in range(self.hiddenLayer):
                self.weightHidden[i][j] -= (self.learningRate * (hidden[j] * x[i]))
                self.biasHidden[j] -= (self.learningRate * hidden[j])
    
    def showError(self, epoch, error):
        print("Epoch: ", epoch, " Error: ", error)

    # Returns the predictions for ever element of input x
    def predict(self, x, y):
        predictions = []
        # Forward propagation
        forward = np.matmul(x, self.weightHidden) + self.biasHidden
        forward = np.matmul(forward, self.weightOutput) + self.biasOutput

        for i in forward:
            predictions.append(np.argmax(i))

        score = []
        for i in range(len(predictions)):
            if predictions[i] == 0: 
                score.append([i, 'Iris-setosa', predictions[i], y[i]])
            elif predictions[i] == 1:
                 score.append([i, 'Iris-versicolour', predictions[i], y[i]])
            elif predictions[i] == 2:
                 score.append([i, 'Iris-virginica', predictions[i], y[i]])
        
        data = pd.DataFrame(score, columns=['Index', 'Predicted', 'Score', 'Actual'])
        return predictions, data
    
    def fit(self, x, y):
        epochCount = 1
        error = 0
        n = len(x);
        epochArr = []
        errorArr = []
        w0 = []
        w1 = []
        while(epochCount <= self.maxEpochs):
            for index, inputs in enumerate(x):
                # Forward propagation
                forward = np.matmul(inputs, self.weightHidden) + self.biasHidden
                self.outputL1 = self.activation(forward)
                forward = np.matmul(self.outputL1, self.weightOutput) + self.biasOutput
                self.outputL2 = self.activation(forward)

                # Backpropagation
                self.Backpropagation_Algorithm(inputs)
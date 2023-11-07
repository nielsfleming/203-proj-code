import seaborn as sns
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.base import BaseEstimator, ClassifierMixin

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
    l1 = dataset[0:40]
    t1 = dataset[40:50]
    l2 = dataset[50:90]
    t2 = dataset[90:100]
    l3 = dataset[100:140]
    t3 = dataset[140:150]
    train = np.concatenate((l1, l2, l3))
    test = np.concatenate((t1, t2, t3))
    return train, test

train_files, test_files = separate_data()

train_x = np.array([i[:4] for i in train_files])
train_y = np.array([i[4] for i in train_files])
test_x = np.array([i[:4] for i in test_files])
test_y = np.array([i[4] for i in test_files])

class MLP(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.inputLayer = 4 # We have 4 different types of data to feed as inputs
        self.hiddenLayer = 5
        self.outputLayer = 3 # There are 3 clasfications for this dataset
        self.learningRate = 0.005
        self.maxEpochs = 1000
        self.biasHiddenValue = -1
        self.biasOutputValue = -1
        self.activation = self.activationFunction
        self.derivative = self.derivativeFunction

        # Starting weights and biases
        ##PROBELM HERE
        self.weightHidden = self.starting_weights(self.hiddenLayer, self.inputLayer)
        self.weightOutput = self.starting_weights(self.outputLayer, self.hiddenLayer)
        self.biasHidden = np.array([self.biasHiddenValue for i in range(self.hiddenLayer)])
        self.biasOutput = np.array([self.biasOutputValue for i in range(self.outputLayer)])
        self.numClasses = 3 

    pass

    def starting_weights(self, x, y):
        return [[2  * random.random() - 1 for i in range(x)] for j in range(y)]
    
    def activationFunction(self, x):
        return 1 / (1 + np.exp(-x))

    def derivativeFunction(self, x):
        return x * (1 - x)

    def Backpropagation_Algorithm(self, x):
        # Error for output layer
        output = []
        errOutput = self.output - self.outputL2
        output = ((-1)*errOutput) * self.derivative(self.outputL2)

        # Updating weights in output layer and hidden layer
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
        totalError = 0
        n = len(x);
        epochArr = []
        errorArr = []
        w0 = []
        w1 = []

    
        while(epochCount <= self.maxEpochs):
            for index, inputs in enumerate(x):
                self.output = np.zeros(self.numClasses)

                # Forward propagation
                self.outputL1 = self.activation((np.dot(inputs, self.weightHidden) + self.biasHidden.T))
                self.outputL2 = self.activation((np.dot(self.outputL1, self.weightOutput) + self.biasOutput.T))

                if(y[index] == 0): 
                    self.output = np.array([1,0,0]) #Class1 {1,0,0}
                elif(y[index] == 1):
                    self.output = np.array([0,1,0]) #Class2 {0,1,0}
                elif(y[index] == 2):
                    self.output = np.array([0,0,1]) #Class3 {0,0,1}

                squareError = 0
                for i in range(self.outputLayer):
                    originalError = (self.output[i] - self.outputL2[i])**2
                    squareError = (squareError + (0.05 * originalError))
                    totalError = totalError + squareError

                # Backpropagation to update weights
                self.Backpropagation_Algorithm(inputs)
            
            totalError = (totalError / n)
            if((epochCount % 50) == 0) or (epochCount == 1):
                self.showError(epochCount, totalError)
                epochArr.append(epochCount)
                errorArr.append(totalError)
            w0.append(self.weightHidden)
            w1.append(self.weightHidden)
            
            epochCount += 1
        
        plt.plot(w0[0])
        plt.title('hidden layer weight update during training')
        plt.legend(['neuron1', 'neuron2', 'neuron3', 'neuron4', 'neuron5'])
        plt.ylabel('Value Weight')
        plt.show()
        
        plt.plot(w1[0])
        plt.title('output layer weight update during training')
        plt.legend(['neuron1', 'neuron2', 'neuron3'])
        plt.ylabel('Value Weight')
        plt.show()

        return self
        
MLPTest = MLP()
MLPTest.fit(train_x, train_y)

previous, data = MLPTest.predict(test_x, test_y)
hits = setosa = versicolour = virginica = 0
numSetosa = numVersicolour = numVirginica = 0
for i in range(len(test_y)):
    if(test_y[i] == 0): numSetosa += 1
    elif(test_y[i] == 1): numVersicolour += 1
    elif(test_y[i] == 2): numVirginica += 1

for i in range(len(test_y)):
    ###### may have an issue here with the if statements
    if(test_y[i] == previous[i]):
        hits += 1
    if(test_y[i] == previous[i] and test_y[i] == 0): 
        setosa += 1
    elif(test_y[i] == previous[i] and test_y[i] == 1): 
        versicolour += 1
    elif(test_y[i] == previous[i] and test_y[i] == 2): 
        virginica += 1

hits = (hits / len(test_y)) * 100
faults = 100 - hits

data

hitsGraph = []
print("Percents :","%.2f"%(hits),"% hits","and","%.2f"%(faults),"% faults")
print("Total samples of test",samples)
print("Iris-Setosa:",numSetosa,"samples")
print("Iris-Versicolour:",numVersicolour,"samples")
print("Iris-Virginica:",numVirginica,"samples")

hitsGraph.append(hits)
hitsGraph.append(faults)
labels = 'Hits', 'Faults'
sizes = [95, 3]
explode = (0, 0.14)

fig1, ax1 = plt.subplots()
ax1.pie(hitsGraph,explode=explode,colors=['green', 'red'], autopct='%1.1f%%',labels=labels,shadow=True,startangle=90)
ax1.axis('equal')
plt.show()
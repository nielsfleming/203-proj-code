import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# Neural Network Structure
def initialize_parameters(input_size, hidden_size, output_size):
    # initializing weights and biases
    W1 = np.random.randn(hidden_size, input_size) * 0.01
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.randn(output_size, hidden_size) * 0.01
    b2 = np.zeros((hidden_size, 1))

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters


# Activation Functions and Their Derivatives
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def relu_derivative(z):
    return (z > 0).astype(float)


# Implementing Forward Propagation
def forward_propagation(X, parameters):
    W1, b1, W2, b2 = parameters['W1'], parameters['b1'], parameters['W2'], parameters['b2']

    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache


# Compute Loss
def compute_mse_loss(Y_true, Y_pred):
    m = Y_true.shape[1]
    loss = (1/m) * np.sum((Y_true - Y_pred) ** 2)
    return loss

def compute_cross_entropy_loss(Y_true, Y_pred):
    m = Y_true.shape[1]
    loss = -(1/m) * np.sum(Y_true *np.log(Y_pred) + (1 - Y_true) * np.log(1 - Y_pred))
    return loss


# Backwards Propagation
def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    W1, W2 = parameters['W1'], parameters['W2']
    A1, A2 = cache['A1'], cache['A2']

    dZ2 = A2 - Y
    dW2 = 1/m * np.dot(dZ2, A1.T)
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * relu_derivative(cache('Z1'))
    dW1 = 1/m * np.dot(dZ1, X.T)
    db1 = 1/m * np.sum(dZ1, axiz=1, keepdims=True)

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return grads


# Updating Parameters
def update_parameters(parameters, grads, learning_rate=0.01):
    W1, b1, W2, b2 = parameters['W1'], parameters['b1'], parameters['W2'], parameters['b2']
    
    dW1, db1, dW2, db2 = grads['dW1'], grads['db1'], grads['dW2'], grads['db2']
    
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters

# Loading and Preprocessing Data    
mnist = fetch_openml('mnist_784')
X, Y = mnist['data'], mnist['target']

# We have to normalize the pixel values
X = X / 255.0

# Convert labels to integers and then to one-hot encoding
Y = Y.astype(int)
Y_one_hot = np.eye(10)[Y].T


# Splitting the Data
split = 60000
X_train, X_test = X[:split].T, X[split:].T
Y_train, Y_test = Y_one_hot[:,:split], Y_one_hot[:,split:]


# Initialize Parameters
parameters = initialize_parameters(784, 128, 10)

# Training the Neural Network :)
epochs = 1000
learning_rate = 0.1
for i in range(epochs):
    # Forward propagation
    A2, cache = forward_propagation(X_train, parameters)
    
    # Compute loss
    loss = compute_cross_entropy_loss(Y_train, A2)
    
    # Backward propagation
    grads = backward_propagation(parameters, cache, X_train, Y_train)
    
    # Update parameters
    parameters = update_parameters(parameters, grads, learning_rate)
    
    if i % 100 == 0:
        print(f"Epoch {i}, Loss: {loss}")


# Model Evaluation

# Forward propagation on the test set
A2, _ = forward_propagation(X_test, parameters)

# Convert predictions to labels
predictions = np.argmax(A2, axis=0)
labels = np.argmax(Y_test, axis=0)

# Compute accuracy
accuracy = np.mean(predictions == labels)
print(f"Test Accuracy: {accuracy * 100:.2f}%")


# To Help Visualize Predictions
index = np.random.randint(X_test.shape[1])
plt.imshow(X_test[:, index].reshape(28, 28), cmap='gray')
plt.title(f"Predicted: {predictions[index]}, True: {labels[index]}")
plt.show()
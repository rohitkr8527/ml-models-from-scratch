import numpy as np
import pandas as pd

# Load CSV 
def load_csv(path):
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].values  # features
    y = df.iloc[:, -1].values   # label (0 or 1)
    return X, y

# Normalization function
def normalize(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)

# Add bias term (intercept)
def add_bias(X):
    n = X.shape[0]
    ones = np.ones((n, 1))     
    return np.hstack((ones, X))  

# Sigmoid function 
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# binary cross-entropy loss
def loss(y, p):
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

# gradient descent
def train(X, y, lr=0.1, epochs=1000):
    X = add_bias(X)              
    n, m = X.shape               
    w = np.zeros(m)              # initialize weights (including bias)

    for _ in range(epochs):
        z = np.dot(X, w)        
        p = sigmoid(z)           
        grad = np.dot(X.T, p - y) / n  
        w -= lr * grad           

    return w

# Predict function
def predict(X, w, thresh=0.5):
    X = add_bias(X)
    p = sigmoid(np.dot(X, w))
    return (p >= thresh).astype(int)

# accuracy 
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


if __name__ == "__main__":
    path = "data.csv"   # put your CSV path here
    X, y = load_csv(path)
    X = normalize(X)

    weights = train(X, y, lr=0.1, epochs=1000)
    preds = predict(X, weights)

    print("Weights (bias + features):", weights)
    print("Accuracy:", accuracy(y, preds))

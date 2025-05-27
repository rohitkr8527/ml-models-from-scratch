import numpy as np
import pandas as pd

# Load csv
def load_csv(file_path):
    data = pd.read_csv(file_path)
    X = data.iloc[:, :-1].values   # features
    y = data.iloc[:, -1].values    # target 
    return X, y

# Normalization function
def normalize(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)

# bias term addition
def add_bias(X):
    n_samples = X.shape[0]
    bias = np.ones((n_samples, 1))  
    return np.hstack((bias, X))     

# gradient descent
def train_ridge(X, y, learning_rate=0.01, epochs=2000, alpha=0.1):
    X = add_bias(X)                 
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)  # initialize weights (including bias)

    for _ in range(epochs):
        predictions = np.dot(X, weights)           
        errors = predictions - y                    

        # Gradient calculation:
        # For bias term (weights[0]) no regularization
        # For other weights, add 2 * alpha * weight to gradient
        grad = (np.dot(X.T, errors) / n_samples) + 2 * alpha * np.r_[0, weights[1:]]

        weights -= learning_rate * grad            

    return weights

# predict function
def predict(X, weights):
    X = add_bias(X)
    return np.dot(X, weights)

# mse function
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


if __name__ == "__main__":
    file_path = "data.csv"        # replace with your CSV file path
    X, y = load_csv(file_path)
    X = normalize(X)                   

    weights = train_ridge(X, y, learning_rate=0.01, epochs=2000, alpha=0.1)
    preds = predict(X, weights)

    print("Learned weights (including bias):", weights)
    print("Mean Squared Error:", mean_squared_error(y, preds))


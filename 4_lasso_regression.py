import numpy as np
import pandas as pd

def load_csv(fp):
    data = pd.read_csv(fp)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y

def normalize(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)

def add_bias(X):
    n = X.shape[0]
    b = np.ones((n, 1))
    return np.hstack((b, X))

def train_lasso(X, y, lr=0.01, epochs=2000, alpha=0.1):
    X = add_bias(X)
    n, m = X.shape
    w = np.zeros(m)

    for _ in range(epochs):
        pred = np.dot(X, w)
        err = pred - y

        grad = np.dot(X.T, err) / n
        grad[1:] += alpha * np.sign(w[1:])  # L1 penalty, no regularization on bias

        w -= lr * grad

    return w

def predict(X, w):
    X = add_bias(X)
    return np.dot(X, w)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

if __name__ == "__main__":
    fp = "data.csv"      # replace with your CSV file path
    X, y = load_csv(fp)
    X = normalize(X)

    w = train_lasso(X, y, lr=0.01, epochs=2000, alpha=0.1)
    preds = predict(X, w)

    print("Weights (including bias):", w)
    print("MSE:", mse(y, preds))

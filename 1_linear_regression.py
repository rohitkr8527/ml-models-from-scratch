import numpy as np
import pandas as pd

# load data
def load_csv(path):
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].values  # features
    y = df.iloc[:, -1].values   # target
    return X, y

# normalization
def normalize(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)

# add bias term (intercept)
def add_bias(X):
    return np.column_stack((np.ones(X.shape[0]), X))

# gradient descent
def fit(X, y, lr=0.01, epochs=1000):
    X = add_bias(X)
    n, m = X.shape
    w = np.zeros(m)

    for _ in range(epochs):
        preds = X @ w
        error = preds - y
        grad = (X.T @ error) / n
        w -= lr * grad

    return w

# predict function
def predict(X, w):
    X = add_bias(X)
    return X @ w

# mean squared error
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


if __name__ == "__main__":
    file_path = "  "  # <- add your CSV file path here
    X, y = load_csv(file_path)
    X = normalize(X)
    weights = fit(X, y, lr=0.01, epochs=1000)
    y_pred = predict(X, weights)

    print("weights:", weights)
    print("mse:", mse(y, y_pred))

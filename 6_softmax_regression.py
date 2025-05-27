import numpy as np
import pandas as pd

def load_csv(file):
    data = pd.read_csv(file)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y

def normalize(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)

def add_bias(X):
    n = X.shape[0]
    b = np.ones((n, 1))       # bias term (intercept)
    return np.hstack((b, X))

def softmax(z):
    # Softmax turns logits into probabilities that sum to 1
    e = np.exp(z - np.max(z, axis=1, keepdims=True))  
    return e / e.sum(axis=1, keepdims=True)

def one_hot(y, c):
    # One-hot encoding for classes (needed for vectorized loss)
    oh = np.zeros((y.size, c))
    oh[np.arange(y.size), y] = 1
    return oh

def train_sm(X, y, lr=0.1, epochs=1000):
    X = add_bias(X)
    n, m = X.shape
    k = len(np.unique(y))    
    y_enc = one_hot(y, k)

    w = np.zeros((m, k))     

    for _ in range(epochs):
        z = np.dot(X, w)         
        p = softmax(z)          

        err = p - y_enc          
        grad = np.dot(X.T, err) / n

        w -= lr * grad           

    return w

def predict(X, w):
    X = add_bias(X)
    z = np.dot(X, w)
    p = softmax(z)
    return np.argmax(p, axis=1)  # class with highest prob

def acc(y_true, y_pred):
    return np.mean(y_true == y_pred)

if __name__ == "__main__":
    fp = "data_multiclass.csv"
    X, y = load_csv(fp)
    X = normalize(X)

    w = train_sm(X, y, lr=0.1, epochs=1000)
    preds = predict(X, w)

    print("Accuracy:", acc(y, preds))
    print("Weights (bias + features):", w)

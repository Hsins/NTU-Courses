import sys
import numpy as np
import pandas as pd

# Constant
lRate = 0.05
iterTime = 3000

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def readFile(filename):
    return pd.read_csv(filename).as_matrix().astype('float')
    
I = [0, 1, 3, 4, 5]
def regulate(X):
    return np.concatenate((X, X[:, I] ** 2, X[:, I] ** 3, X[:, I] ** 4, X[:, I] ** 5, np.log(X[:, I] + 1e-10), (X[:, 0] * X[:, 3]).reshape(X.shape[0], 1), (X[:, 0] * X[:, 5]).reshape(X.shape[0], 1), (X[:, 0] * X[:, 5]).reshape(X.shape[0], 1) ** 2, (X[:, 3] * X[:, 5]).reshape(X.shape[0], 1), X[:, 6:] * X[:, 5].reshape(X.shape[0], 1), (X[:, 3] - X[:, 4]).reshape(X.shape[0], 1), (X[:, 3] - X[:, 4]).reshape(X.shape[0], 1) ** 3), axis = 1)

X_train, Y_train, X_test = readFile(sys.argv[1]), readFile(sys.argv[2]), readFile(sys.argv[3])
X_train, X_test = regulate(X_train), regulate(X_test)

NUM = 6000
X_train, Y_train = X_train[:-NUM], Y_train[:-NUM]
valid = (X_train[-NUM:], Y_train[-NUM:])
meanX, stdX = np.mean(X_train, axis = 0), np.std(X_train, axis = 0)

def cost(X, y, w):
    pred = sigmoid(np.dot(X, w))
    return -np.mean(y * np.log(pred + 1e-20) + (1 - y) * np.log((1 - pred + 1e-20)))

def scale(X, mean, std):
    return (X - mean) / (std + 1e-20)

def evaluate(X, y, w):
    p = sigmoid(np.dot(X, w))
    p[p < 0.5] = 0.0
    p[p >= 0.5] = 1.0
    return np.mean(1 - np.abs(y - p))

def regression(X, y, LR, iters, C):
    X = scale(X, meanX, stdX)                                                       # Normalize
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis = 1)                     # Add bias
    w = np.zeros((X.shape[1], 1))   

    X_valid = scale(valid[0], meanX, stdX)                                          # Normalize                                       
    X_valid = np.concatenate((np.ones((X_valid.shape[0], 1)), X_valid), axis = 1)   # Add bias
    Y_valid = valid[1]
    
    initLR = LR                                                                     # Save the original LearnRate 
    wLR = 0.0

    for i in range(iters):
        pred = sigmoid(np.dot(X, w))
        grad = -np.dot(X.T, (y - pred))

        wLR = wLR + grad ** 2
        LR = initLR / np.sqrt(wLR)
        w = w - LR * (grad + C * np.sum(w))

        if i % 100 == 0:
            print('[Iters {:5d}] - training cost: {:.5f}, accuracy: {:.5f}'.format(i, cost(X, y, w), evaluate(X, y, w)))
            print('\tvalid cost: {:.5f}, accuracy: {:.5f}'.format(cost(X_valid, Y_valid, w), evaluate(X_valid, Y_valid, w)))
    return w

w = regression(X_train, Y_train, lRate, iterTime, 0.0)

with open(sys.argv[4], 'w') as fout:
    print('id,label', file = fout)
    X_test = scale(X_test, meanX, stdX)                                             # Normalize
    X_test = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), axis = 1)      # Add bias
    pred = sigmoid(np.dot(X_test, w))
    for (i, v) in enumerate(pred.flatten()):
        print('{},{}'.format(i + 1, 1 if v >= 0.5 else 0), file = fout)

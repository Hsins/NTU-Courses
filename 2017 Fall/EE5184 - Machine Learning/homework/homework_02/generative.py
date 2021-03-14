import sys
import numpy as np
import pandas as pd

def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-14, 1-(1e-14))

def readFile(filename):
    return pd.read_csv(filename).as_matrix().astype('float')

X_train, Y_train, X_test = readFile(sys.argv[1]), readFile(sys.argv[2]), readFile(sys.argv[3])

meanX, stdX = np.mean(X_train, axis = 0), np.std(X_train, axis = 0)
meanX, stdX = 0, 1
nums, mus = [], []

def scale(X, mean, std): 
    return (X - mean) / (std + 1e-20)

def evaluate(X, y, invCov):
    pred = predict(X, invCov)
    pred = np.around(1.0 - pred)
    result = (y.flatten() == pred)
    return np.mean(result)

def predict(X, invCov):
    X = scale(X, meanX, stdX)
    W = np.dot((mus[0] - mus[1]), invCov)
    X = X.T
    print(invCov)
    B = (-0.5) * np.dot(np.dot(mus[0], invCov), mus[0]) + (0.5) * np.dot(np.dot([mus[1]], invCov), mus[1]) + np.log(nums[0] / nums[1])
    a = np.dot(W, X) + B
    y = sigmoid(a)
    # print(y)
    return y

def generative(X, y, CLASS = 2):
    X = scale(X, meanX, stdX)

    shareCov = 0.0
    for i in range(CLASS):
        C = X[(y == i).flatten()]
        num = C.shape[0]
        mu = np.mean(C, axis = 0)
        cov = np.mean([(C[i] - mu).reshape((-1, 1)) * (C[i] - mu).reshape((1, -1)) for i in range(C.shape[0])], axis = 0)
        nums.append(num)
        mus.append(mu)
        shareCov += (num / X.shape[0]) * cov
    
    invCov = np.linalg.inv(shareCov)
    print('training accuracy: {:.5f}'.format(evaluate(X, y, invCov)))
    return invCov
   
invCov = generative(X_train, Y_train, 2)

with open(sys.argv[4], 'w') as fout:
    print('id,label', file = fout)
    pred = predict(X_test, invCov)
    for (i, v) in enumerate(pred.flatten()):
        print('{},{}'.format(i + 1, 0 if v >= 0.5 else 1), file = fout)

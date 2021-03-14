import numpy as np
import pandas as pd
import math
import sys

# Constant
# Note the indexs of each items:
#  1: AMB_TEMP    10: PM2.5
#  2: CH4         11: RAINFALL
#  3: CO          12: RH
#  4: NMHC        13: SO2
#  5: NO          14: THC
#  6: NO2         15: WD_HR
#  7: NOx         16: WIND_DIREC
#  8: O3          17: WIND_SPEED
#  9: PM10        18: WS_HR
MONTH, HOUR = 11, 9
FEA, INDEXofPM25 = 3, 1
O3, PM25, SO2 = 8, 10, 13   # O3, PM2.5, SO2

# Initialize learning rate, iterate times and regulation parameter
LEARNRATE = 100
ITERATE = 100000
LAMBDA = 0.0

df = pd.read_csv('data/train.csv', header = None, encoding = 'Big5')
df = df.drop(df.index[0])
df = df.drop(df.columns[[0, 1, 2]], axis = 1)

if MONTH == 11:
    df = df.drop(df.index[2160:2520])        

if FEA == 3:
    for i in range(18):
        if i == O3 or i == PM25 or i == SO2:
            continue
        df = df.drop(df.index[df.index % 18 == i])

df = np.matrix(df.values)
df[df == 'NR'] = 0

# Initialize 1/1 (train.csv)
data = df[0: FEA, :]
for i in range(1, MONTH * 20):
    data = np.concatenate((data, df[i * FEA:i * FEA + FEA, :]), axis = 1)

def assignValue(hour, INDEXofPM25):
    X = np.zeros(((480 - hour) * MONTH, hour * FEA))
    y = np.zeros(((480 - hour) * MONTH, 1))
    j = 0      
    for i in range(data.shape[1]):
        if i % 480 > 480 - hour - 1:
            continue
        X[j] = data[: , i: i + hour].reshape(1, FEA * hour)
        y[j] = data[INDEXofPM25, i + hour]
        j = j + 1
    return X, y

X, y = assignValue(HOUR, INDEXofPM25)

X = np.concatenate((X, X ** 2), axis = 1)                       # Add square term
X = np.concatenate((np.ones((X.shape[0], 1)), X), axis = 1)     # Add bias

w = np.zeros((X.shape[1], 1))                                   # Initialize weight
Sgra = np.zeros((X.shape[1], 1))                                # Initialize Sgra

# Train the model
for i in range(ITERATE):
    loss = np.dot(X, w) - y 
    cost = (np.sum(loss ** 2) + LAMBDA * np.sum(w ** 2)) / X.shape[0]
    costA  = math.sqrt(cost)
    gra = np.dot(X.T, loss)
    Sgra += gra ** 2
    ada = np.sqrt(Sgra)
    w = w - LEARNRATE * gra / ada
    print ('iteration: %d | Cost: %f  ' % (i, costA))

np.save('model.npy', w)
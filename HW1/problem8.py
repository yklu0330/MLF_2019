import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import random

def sign(z):
    if z > 0:
        return 1
    else:
        return -1

def dot(w, y, x):
    new = np.zeros(len(w))
    for i in range(len(w)):
        new[i] = w[i] + y[0] * x[i]
    return new

def errCal(data, w, wNew):
    wErr = 0
    wNewErr = 0
    dataVariety = data['variety'].unique()
    for index, row in data.iterrows():
        arr = np.array(row[:randomData.shape[1] - 1])
        x = np.concatenate((np.array([1.]), arr))
        y = 0
        if row['variety'] == dataVariety[0]:
            y = np.array([1.])
        elif row['variety'] == dataVariety[1]:
            y = np.array([-1.])
        if sign(np.dot(w, x)) != y:
            wErr = wErr + 1
        if sign(np.dot(wNew, x)) != y:
            wNewErr = wNewErr + 1
    if wErr < wNewErr:
        return 1
    else:
        return 0

trainData = pd.read_csv('hw1_7_train.dat', names=['a', 'b', 'c', 'd', 'variety'], skiprows=0, header=None, sep='\s+')
testData = pd.read_csv('hw1_7_test.dat', names=['a', 'b', 'c', 'd', 'variety'], skiprows=0, header=None, sep='\s+')

dataVariety = trainData['variety'].unique()

errorRate = []
errRateMean = 0
updateFreq = []

for i in range(1126):
    randomData = trainData.sample(frac=1)
    datasize = randomData.shape[0]

    w = np.zeros((trainData.shape[1]))
    wPocket = np.zeros((trainData.shape[1]))

    update = 0

    for index, row in randomData.iterrows():
        arr = np.array(row[:randomData.shape[1] - 1])
        x = np.concatenate((np.array([1.]), arr))
        if row['variety'] == dataVariety[0]:
            y = np.array([1.])
        elif row['variety'] == dataVariety[1]:
            y = np.array([-1.])
        if sign(np.dot(w, x)) != y:
            w = dot(w, y, x)
            wPocket = w
        update += 1
        if update > 100:
            break;

    updateFreq.append(update)

    error = 0
    for index, row in testData.iterrows():
        arr = np.array(row[:testData.shape[1] - 1])
        x = np.concatenate((np.array([1.]), arr))
        y = 0
        if row['variety'] == dataVariety[0]:
            y = np.array([1.])
        elif row['variety'] == dataVariety[1]:
            y = np.array([-1.])

        predict = sign(np.dot(wPocket, x))
        if predict != y:
            error += 1
    errorRate.append(error / testData.shape[0])
    errRateMean = errRateMean + error / testData.shape[0]
    print(error / testData.shape[0])

print('error rate: %f' %(errRateMean / 1126))

plt.xlabel("error rate")
plt.ylabel("frequency")
plt.hist(errorRate, bins='auto')
plt.show()

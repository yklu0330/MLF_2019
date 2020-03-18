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
    for i in range(len(w)):
        w[i] = w[i] + 0.5 * y[0] * x[i]
    return w


# load vertebral dataset from csv
dataset = pd.read_csv('hw1_6_train.dat', names=['a', 'b', 'c', 'd', 'variety'], skiprows=0, header=None, sep='\s+')
# for index, row in dataset.iterrows():
#     print(row[:])

dataVariety = dataset['variety'].unique()

updateTotal = 0
updateFreq = []

for i in range(1126):
    randomData = dataset.sample(frac=1)
    datasize = randomData.shape[0]

    w = np.zeros((dataset.shape[1]))

    error = 1
    iterator = 0

    while error != 0:
        error = 0
        for index, row in randomData.iterrows():
            arr = np.array(row[:randomData.shape[1] - 1])
            x = np.concatenate((np.array([1.]), arr))
            if row['variety'] == dataVariety[0]:
                y = np.array([1.])
            elif row['variety'] == dataVariety[1]:
                y = np.array([-1.])
            if sign(np.dot(w, x)) != y:
                w = dot(w, y, x)
                error += 1
                iterator += 1
    updateFreq.append(iterator)
    updateTotal = updateTotal + iterator
    print(iterator)


print(updateTotal / 1126)

plt.xlabel("number of updates")
plt.ylabel("frequency")
plt.hist(updateFreq, bins='auto')
plt.show()

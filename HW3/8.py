import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import random
import wget
import math

def sigmoid(x):
	return 1 / (1 + math.exp(-x))

def sign(x):
	if x >= 0:
		return 1
	else:
		return -1

# trainURL = "https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_algo/hw3_train.dat"
# wget.download(trainURL, out='./hw3_train.dat')
# testURL = "https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_algo/hw3_test.dat"
# wget.download(testURL, out='./hw3_test.dat')

learnRate1 = 0.01
iteration = 2000

trainDS = pd.read_csv('hw3_train.dat', skiprows=0, header=None, sep='\s+')
trainLabel = trainDS[20].values
droptrainDS = trainDS.drop(20, axis=1)
droptrainData = droptrainDS.values
addOne1 = np.ones([trainDS.shape[0], 1])
trainData = np.concatenate((addOne1, droptrainData), axis=1)

testDS = pd.read_csv('hw3_test.dat', skiprows=0, header=None, sep='\s+')
testLabel = testDS[20].values
droptestDS = testDS.drop(20, axis=1)
droptestData = droptestDS.values
addOne2 = np.ones([testDS.shape[0], 1])
testData = np.concatenate((addOne2, droptestData), axis=1)

w1 = np.zeros([trainDS.shape[1], 1])

gradEout = []
for i in range(iteration):
	grad = np.zeros([trainDS.shape[1], 1])
	# calculate batch gradient
	for j in range(trainDS.shape[0]):
		xn = trainData[j].reshape(trainData.shape[1], -1)
		temp = -1 * float(trainLabel[j]) * np.dot(w1.T, xn)
		grad += -1 * sigmoid(temp) * trainLabel[j] * xn
	grad /= trainDS.shape[0]
	w1 -= learnRate1 * grad

	# calculate Eout
	Eout = 0
	for j in range(testDS.shape[0]):
		xn = testData[j].reshape(testData.shape[1], -1)
		if sign(np.dot(w1.T, xn)) != testLabel[j]:
			Eout += 1
	Eout /= testDS.shape[0]
	gradEout.append(Eout)
	print("gradEout: %f" %(Eout))

learnRate2 = 0.001
w2 = np.zeros([trainDS.shape[1], 1])
stochEout = []
i = 0
it = 0
while i < trainDS.shape[0]:
	# calculate stochastic gradient
	stoch_grad = np.zeros([trainDS.shape[1], 1])
	xn = trainData[i].reshape(trainData.shape[1], -1)
	temp = -1 * float(trainLabel[i]) * np.dot(w2.T, xn)
	stoch_grad += -1 * sigmoid(temp) * trainLabel[i] * xn
	w2 -= learnRate2 * stoch_grad
	
	# calculate Eout
	Eout = 0
	for j in range(testDS.shape[0]):
		xn = testData[j].reshape(testData.shape[1], -1)
		if sign(np.dot(w2.T, xn)) != testLabel[j]:
			Eout += 1
	Eout /= testDS.shape[0]
	stochEout.append(Eout)
	print("stochEout: %f" %(Eout))

	i += 1
	if i == trainDS.shape[0]:
		i = 0
	it += 1
	if it >= iteration:
		break

iterList = []
for i in range(iteration):
	iterList.append(i+1)

plt.plot(iterList, gradEout, color = 'r', label="gradient", linewidth = 2)
plt.plot(iterList, stochEout, color = 'b', label="stochastic gradient", linewidth = 2)
plt.xlabel("iteration", fontsize=12)
plt.ylabel("Eout", fontsize=12)
plt.legend(loc = "best", fontsize=12)
plt.show()


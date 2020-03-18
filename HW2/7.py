import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


def sign(z):
	if z >= 0:
		return 1
	else:
		return -1

dataSize = 20
noiseProb = 0.2
theta = 0
iteration = 1000

avgEin = 0
avgEout = 0
dif = []

for iter in range(iteration):
	random.seed(datetime.now())
	x = np.random.uniform(-1, 1, dataSize)
	sortX = np.sort(x)
	y = np.zeros(sortX.shape)

	for i in range(dataSize):
		y[i] = sign(sortX[i])
		noise = random.random()
		if noise <= 0.2:
			y[i] = -y[i]

	minEin = 1.0
	bestTheta = 0.0
	bestS = 0.0

	for k in range(2):
		if k == 0:
			s = 1
		else:
			s = -1
		for i in range(dataSize + 1):
			if i == 0:
				theta = sortX[0] - 1
			elif i == dataSize:
				theta = sortX[dataSize-1] + 1
			else:
				theta = (sortX[i-1] + sortX[i]) / 2

			error = 0.0
			for j in range(dataSize):
				fx = s * sign(sortX[j] - theta)
				if fx != y[j]:
					error += 1
			errorRate = error / dataSize
			
			if errorRate < minEin:
				minEin = errorRate
				bestTheta = theta
				bestS = s

	Eout = 0.5 + 0.3 * bestS * (abs(bestTheta) - 1)

	print("%d %f %f" %(iter, minEin, Eout))

	dif.append(minEin - Eout)

	avgEin += minEin
	avgEout += Eout

avgEin /= 1000
avgEout /= 1000

print(avgEin)
print(avgEout)


plt.xlabel("Ein - Eout")
plt.ylabel("frequency")
plt.hist(dif, bins='auto')
plt.show()








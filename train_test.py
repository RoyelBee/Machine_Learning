import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd

# Populate the data set
np.random.seed(2)
pageSpeeds = np.random.normal(3.0, 1.0, 100)
purchaseAmount = np.random.normal(50.0, 30.0, 100) / pageSpeeds

# plt.scatter(pageSpeeds, purchaseAmount)
# plt.xlabel('Page Speeds')
# plt.ylabel('Purchase Amount')
# plt.show()

# Split train and test data
trainX = pageSpeeds[:80]
testX = pageSpeeds[80:]

trainY = purchaseAmount[:80]
testY = purchaseAmount[80:]

# Plot those the data how it looks like
plt.scatter(trainX, trainY)
plt.scatter(testX, testY)
# plt.show()

# Now use 8th degree polynomial to test over fitting
x = np.array(trainX)
y = np.array(trainY)
p8 = np.poly1d(np.polyfit(x, y, 8))

# Lest plot the train data
ax = np.linspace(0, 7, 100)
axes = plt.axes()
axes.set_xlim([0, 7])
axes.set_ylim([0, 200])
# plt.scatter(x, y)
plt.plot(ax, p8(ax), c='r')
# plt.show()

# Now plot test data
testX = np.array(testX)
testY = np.array(testY)
axes = plt.axes()
axes.set_xlim([0, 7])
axes.set_ylim([0, 200])
plt.scatter(testX, testY)
plt.plot(ax, p8(ax), c='r')
plt.show()

# Now test the model accuracy
from sklearn.metrics import r2_score

r2 = r2_score(testY, p8(testX))
print('Accuracy is = ', r2)

r2 = r2_score(np.array(trainY), p8(np.array(trainX)))
print('New model accuracy = ', r2)


print('Finished ')

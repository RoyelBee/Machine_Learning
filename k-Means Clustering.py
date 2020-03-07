import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt

# Create fake income / Age clusters data for N peoples in k cluster
def createClusteredData(N, k):
    np.random.seed(10)
    pointsPerCluster = float(N)/k
    x = []

    for i in range(k):
        incomeCentroid = np.random.uniform(20000, 200000)
        ageCentroid = np.random.uniform(20, 70)

        for j in range(int(pointsPerCluster)):
            x.append([np.random.normal(incomeCentroid, 10000), np.random.normal(ageCentroid, 2)])
    x = np.array(x)
    return x

data = createClusteredData(100, 5)

model = KMeans(n_clusters=5)
model = model.fit(scale(data))

print(model.labels_)
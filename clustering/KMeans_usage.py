import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X = np.random.rand(100, 2)

model = KMeans(n_clusters=3)
model.fit(X)
labels = model.predict(X)
centroids = model.cluster_centers_

colors = ['r', 'g', 'b']
for i in range(len(X)):
    plt.scatter(X[i, 0], X[i, 1], c=colors[labels[i]])
for centroid in centroids:
    plt.scatter(centroid[0], centroid[1], c='black', marker='x', s=100)
plt.show()

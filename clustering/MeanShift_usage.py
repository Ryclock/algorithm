import numpy as np
from sklearn.cluster import MeanShift
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

data, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=0)

ms = MeanShift(bandwidth=1.6)
ms.fit(data)

labels = ms.labels_
cluster_centers = ms.cluster_centers_
num_clusters = len(np.unique(labels))

plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
            marker='o', s=150, c='red', label='Centroids')
plt.title(f'Mean Shift Clustering with {num_clusters} Clusters')
plt.legend()
plt.show()

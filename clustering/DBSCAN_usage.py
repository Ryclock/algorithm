import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=0)

dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(X)

labels = dbscan.labels_

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=60)
plt.scatter(X[labels == -1][:, 0], X[labels == -1][:, 1], c='red', marker='x', s=100, label='Noise')
plt.title('DBSCAN Clustering')
plt.legend()
plt.show()

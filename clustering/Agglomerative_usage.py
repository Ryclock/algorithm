import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=100, centers=3, cluster_std=1.0, random_state=42)

cluster_model = AgglomerativeClustering(n_clusters=3)
labels = cluster_model.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Agglomerative Clustering Demo')
plt.show()

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

X, y = make_blobs(n_samples=300, centers=4, random_state=42)

gmm = GaussianMixture(n_components=4, random_state=42)
gmm.fit(X)

labels = gmm.predict(X)
means = gmm.means_
covariances = gmm.covariances_

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7)
plt.scatter(means[:, 0], means[:, 1], color='red', marker='x', s=100)
plt.title('GMM Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_blobs


class AgglomerativeClusteringCustom:
    def __init__(self, n_clusters=2, linkage='single'):
        self.n_clusters = n_clusters
        self.linkage = linkage

    def fit_predict(self, X):
        clusters = [[i] for i in range(len(X))]

        while len(clusters) > self.n_clusters:
            min_distance = float('inf')
            merge_indices = (0, 0)

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    distance = self.compute_distance(
                        X[clusters[i]], X[clusters[j]])
                    if distance < min_distance:
                        min_distance = distance
                        merge_indices = (i, j)

            clusters[merge_indices[0]].extend(clusters[merge_indices[1]])
            del clusters[merge_indices[1]]

        labels = np.zeros(len(X), dtype=int)
        for cluster_label, cluster in enumerate(clusters):
            labels[cluster] = cluster_label

        return labels

    def compute_distance(self, cluster1, cluster2):
        if self.linkage == 'single':
            return np.min(np.sqrt(((np.array(cluster1)[:, np.newaxis] - np.array(cluster2)) ** 2).sum(axis=2)))
        elif self.linkage == 'complete':
            return np.max(np.sqrt(((np.array(cluster1)[:, np.newaxis] - np.array(cluster2)) ** 2).sum(axis=2)))
        elif self.linkage == 'average':
            return np.mean(np.sqrt(((np.array(cluster1)[:, np.newaxis] - np.array(cluster2)) ** 2).sum(axis=2)))


if __name__ == "__main__":
    np.random.seed(0)
    X, y = make_blobs(n_samples=150, centers=3,
                      cluster_std=1.0, random_state=42)

    agglomerative_custom = AgglomerativeClusteringCustom(
        n_clusters=3, linkage='single')
    labels_custom = agglomerative_custom.fit_predict(X)

    plt.figure(figsize=(8, 6))

    cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    plt.scatter(X[:, 0], X[:, 1], c=labels_custom,
                cmap=cmap, edgecolors='k', marker='o')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Agglomerative Clustering (Custom)')

    plt.show()

import matplotlib.pyplot as plt
import random


class KMeans:
    def __init__(self, k, max_iterations=100):
        self.k = k
        self.max_iterations = max_iterations

    def fit(self, data):
        self.centroids = self._initialize_centroids(data)
        for _ in range(self.max_iterations):
            clusters = [[] for _ in range(self.k)]
            for point in data:
                centroid_index = self._closest_centroid_index(point)
                clusters[centroid_index].append(point)
            old_centroids = self.centroids[:]
            self.centroids = [self._calculate_centroid(
                cluster) for cluster in clusters]
            if old_centroids == self.centroids:
                break

        return clusters

    def _initialize_centroids(self, data):
        centroids = random.sample(data, self.k)
        return centroids

    def _closest_centroid_index(self, point):
        distances = [self._euclidean_distance(
            point, centroid) for centroid in self.centroids]
        return distances.index(min(distances))

    def _euclidean_distance(self, p1, p2):
        return sum([(p1[i] - p2[i]) ** 2 for i in range(len(p1))]) ** 0.5

    def _calculate_centroid(self, cluster):
        if not cluster:
            return random.choice(self.centroids)
        dimensions = len(cluster[0])
        centroid = [0] * dimensions
        for point in cluster:
            for i in range(dimensions):
                centroid[i] += point[i]
        for i in range(dimensions):
            centroid[i] /= len(cluster)
        return tuple(centroid)


if __name__ == "__main__":
    data = []
    for _ in range(50):
        x = random.uniform(0, 10)
        y = random.uniform(0, 10)
        data.append((x, y))

    model = KMeans(k=3)
    clusters = model.fit(data)

    colors = ['r', 'g', 'b']
    for i, cluster in enumerate(clusters):
        color = colors[i]
        for point in cluster:
            plt.scatter(point[0], point[1], c=color)
        centroid = model.centroids[i]
        plt.scatter(centroid[0], centroid[1], c='black', marker='x', s=100)
    plt.show()

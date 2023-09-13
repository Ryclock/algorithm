def euclidean_distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


def range_query(data, point, epsilon):
    neighbors = []
    for neighbor_point in data:
        if euclidean_distance(point, neighbor_point) <= epsilon:
            neighbors.append(neighbor_point)
    return neighbors


def dbscan(data, epsilon, min_samples):
    clusters = []
    visited = []

    for point in data:
        point_tuple = tuple(point)
        if point_tuple in visited:
            continue
        visited.append(point_tuple)
        neighbors = range_query(data, point, epsilon)

        if len(neighbors) < min_samples:
            continue

        cluster = [point]
        clusters.append(cluster)
        expand_cluster(data, point, neighbors, cluster,
                       epsilon, min_samples, visited)

    return clusters


def expand_cluster(data, point, neighbors, cluster, epsilon, min_samples, visited):
    for neighbor in neighbors:
        if neighbor not in visited:
            visited.append(neighbor)
            new_neighbors = range_query(data, neighbor, epsilon)
            if len(new_neighbors) >= min_samples:
                neighbors.extend(new_neighbors)
        if neighbor not in [p for c in cluster for p in c]:
            cluster.append(neighbor)


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_blobs

    X, _ = make_blobs(n_samples=300, centers=3,
                      cluster_std=0.6, random_state=0)

    epsilon = 3
    min_samples = 5
    clusters = dbscan(X.tolist(), epsilon, min_samples)

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    for i, cluster in enumerate(clusters):
        cluster_points = np.array(cluster)
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    c=colors[i % len(colors)], label=f'Cluster {i + 1}')

    plt.title('DBSCAN Clustering')
    plt.legend()
    plt.show()

import numpy as np


def gaussian_kernel(distance, bandwidth):
    return np.exp(-0.5 * (distance / bandwidth) ** 2)


def mean_shift_clustering(data, bandwidth, num_iterations=100, convergence_threshold=1e-4):
    centers = np.copy(data)

    for _ in range(num_iterations):
        new_centers = np.copy(centers)
        for i in range(len(centers)):
            shift = np.zeros_like(centers[i])
            total_weight = 0.0

            for data_point in data:
                distance = np.linalg.norm(data_point - centers[i])
                if distance < bandwidth:
                    weight = gaussian_kernel(distance, bandwidth)
                    shift += weight * data_point
                    total_weight += weight

            if total_weight > 0:
                new_centers[i] = shift / total_weight

        if np.all(np.abs(new_centers - centers) < convergence_threshold):
            break

        centers = new_centers

    unique_centers = [centers[0]]
    for center in centers:
        is_unique = True
        for unique_center in unique_centers:
            if np.linalg.norm(center - unique_center) < bandwidth / 2:
                is_unique = False
                break
        if is_unique:
            unique_centers.append(center)

    labels = []
    for data_point in data:
        min_distance = float("inf")
        label = -1
        for i, center in enumerate(unique_centers):
            distance = np.linalg.norm(data_point - center)
            if distance < min_distance:
                min_distance = distance
                label = i
        labels.append(label)

    return unique_centers, labels


if __name__ == "__main__":
    np.random.seed(0)
    data = np.random.randn(300, 2)
    data[50:100] += 4
    data[200:250] += 8

    bandwidth = 2.0
    centers, labels = mean_shift_clustering(data, bandwidth)

    import matplotlib.pyplot as plt
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    plt.scatter(np.array(centers)[:, 0], np.array(centers)[
                :, 1], marker='o', s=150, c='red', label='Centroids')
    plt.title('Mean Shift Clustering')
    plt.legend()
    plt.show()

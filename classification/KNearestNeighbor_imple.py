import math


def kNN(X_train, y_train, X_test, k):
    y_pred = []
    for x_test in X_test:
        distances = []
        for i, x_train in enumerate(X_train):
            dist = math.sqrt(
                sum([(x_test[j] - x_train[j]) ** 2 for j in range(len(x_train))]))
            distances.append((i, dist))
        distances.sort(key=lambda x: x[1])
        k_nearest_neighbors_labels = [y_train[i] for i, _ in distances[:k]]
        counts = {}
        for label in k_nearest_neighbors_labels:
            counts[label] = counts.get(label, 0) + 1
        y_pred.append(max(counts, key=lambda x: counts[x]))
    return y_pred


if __name__ == '__main__':
    X_train = [[1, 2], [2, 1], [3, 4], [4, 3]]
    y_train = [0, 0, 1, 1]
    X_test = [[1, 1], [3, 3]]
    y_pred = kNN(X_train, y_train, X_test, 3)
    print(y_pred)

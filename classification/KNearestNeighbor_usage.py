import numpy as np
from sklearn.neighbors import KNeighborsClassifier

if __name__ == '__main__':
    X_train = np.array([[1, 2], [2, 1], [3, 4], [4, 3]])
    y_train = np.array([0, 0, 1, 1])
    X_test = np.array([[1, 1], [3, 3]])
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(y_pred)

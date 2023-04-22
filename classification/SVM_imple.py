import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def svm_loss(X, y, w, C, lamda):
    distances = 1 - y * (np.dot(X, w))
    distances[distances < 0] = 0
    loss = C * (np.sum(distances) / X.shape[0]) + lamda * np.dot(w, w)
    return loss


def svm_gradient(X, y, w, C):
    n = X.shape[0]
    distances = 1 - y * (np.dot(X, w))
    dw = np.zeros(len(w))
    for index, dis in enumerate(distances):
        if max(0, dis) == 0:
            dw += w
            continue
        dw += w - (C / n * y[index] * X[index])
    dw = dw / n
    return dw


def svm_fit(X, y, C, learning_rate, max_epochs, lamda):
    w = np.ones(X.shape[1])
    for epoch in range(1, max_epochs):
        gradient = svm_gradient(X, y, w, C)
        w = w - learning_rate * gradient
        if epoch % 100 == 0:
            loss = svm_loss(X, y, w, C, lamda)
            print(f'Epoch: {epoch}, Loss: {loss}')
    return w


if __name__ == '__main__':
    data = load_breast_cancer()
    X = data['data']
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    intercept = np.ones((X_train_sc.shape[0], 1))
    X_train_sc_ext = np.concatenate((intercept, X_train_sc), axis=1)
    y_train[y_train == 0] = -1
    X_test_sc = scaler.transform(X_test)
    intercept = np.ones((X_test_sc.shape[0], 1))
    X_test_sc_ext = np.concatenate((intercept, X_test_sc), axis=1)

    w = svm_fit(X_train_sc_ext, y_train, C=1,
                learning_rate=0.001, max_epochs=20000, lamda=0.5)

    y_pred = np.dot(X_test_sc_ext, w)
    y_pred[y_pred > 0] = 1
    y_pred[y_pred <= 0] = 0

    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)

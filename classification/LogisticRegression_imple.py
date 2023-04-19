import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def loss(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()


def gradient(X, h, y):
    return np.dot(X.T, (h - y)) / y.shape[0]


def logistic_regression(X, y, num_steps, learning_rate):
    intercept = np.ones((X.shape[0], 1))
    X = np.concatenate((intercept, X), axis=1)
    theta = np.zeros(X.shape[1])

    for i in range(num_steps):
        z = np.dot(X, theta)
        h = sigmoid(z)
        grad = gradient(X, h, y)
        theta -= learning_rate * grad

        if i % 1000 == 0:
            z = np.dot(X, theta)
            h = sigmoid(z)
            print('Loss:', loss(h, y))

    return theta


if __name__ == '__main__':
    data = load_breast_cancer()
    X = data['data']
    y = data['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    X_train_norm = minmax_scale(X_train)
    X_test_norm = minmax_scale(X_test)

    theta = logistic_regression(
        X_train_norm, y_train, num_steps=10000, learning_rate=0.1)
    intercept = np.ones((X_test_norm.shape[0], 1))
    X_test_norm = np.concatenate((intercept, X_test_norm), axis=1)
    y_pred = sigmoid(np.dot(X_test_norm, theta)).round()
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy:', accuracy)

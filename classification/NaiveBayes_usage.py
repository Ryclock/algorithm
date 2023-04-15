import numpy as np
from sklearn.naive_bayes import GaussianNB

if __name__ == '__main__':
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    Y = np.array([0, 0, 0, 1, 1, 1])
    clf = GaussianNB()
    clf.fit(X, Y)
    print(clf.predict([[-0.8, -1], [0.8, 1], [2.8, 1]]))
    print(clf.predict_proba([[-0.8, -1], [0.8, 1], [2.8, 1]]))
    print(clf.predict_log_proba([[-0.8, -1], [0.8, 1], [2.8, 1]]))

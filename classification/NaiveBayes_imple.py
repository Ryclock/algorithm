import numpy as np


class GaussianNB:
    def __init__(self, priors=None, var_smoothing=1e-9) -> None:
        self.priors = priors
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.prior = {}

        for c in self.classes:
            X_c = X[y == c]
            self.mean[str(c)] = np.mean(X_c, axis=0)
            self.var[str(c)] = np.var(X_c, axis=0)
            self.prior[str(c)] = np.shape(X_c)[0] / np.shape(X)[0]

    def predict(self, X):
        posteriors = []

        for c in self.classes:
            posterior = np.prod(self.gaussian_pdf(X, str(c)))
            posteriors.append(posterior*self.prior[str(c)])
        return self.classes[np.argmax(posteriors)]

    def gaussian_pdf(self, X, clazz):
        return (1 / np.sqrt(2*np.pi*self.var[str(clazz)])) * np.exp(-(X-self.mean[str(clazz)])**2 / (2*self.var[str(clazz)]**2))


if __name__ == '__main__':
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    Y = np.array([0, 0, 0, 1, 1, 1])
    clf = GaussianNB()
    clf.fit(X, Y)
    print(clf.predict([-0.8, -1]))
    print(clf.predict([0.8, 1]))
    print(clf.predict([2.8, 1]))

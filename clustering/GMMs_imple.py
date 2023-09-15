import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
X1 = np.random.randn(100, 2) + np.array([5, 5])
X2 = np.random.randn(100, 2) + np.array([10, 10])
X = np.vstack([X1, X2])


class GMM_EM:
    def __init__(self, n_components, n_iterations=100):
        self.n_components = n_components
        self.n_iterations = n_iterations

    def initialize_parameters(self, X):
        self.mu = X[np.random.choice(
            X.shape[0], self.n_components, replace=False)]
        self.sigma = [np.eye(X.shape[1]) for _ in range(self.n_components)]
        self.weights = np.ones(self.n_components) / self.n_components

    def calculate_probabilities(self, X):
        probabilities = np.zeros((X.shape[0], self.n_components))
        for i in range(self.n_components):
            probabilities[:, i] = self.weights[i] * \
                self.multivariate_normal(X, self.mu[i], self.sigma[i])
        return probabilities

    def multivariate_normal(self, X, mu, sigma):
        det = np.linalg.det(sigma)
        inv = np.linalg.inv(sigma)
        N = X.shape[1]
        constant = 1.0 / np.sqrt((2 * np.pi) ** N * det)
        exponent = np.exp(-0.5 * np.sum(np.dot(X - mu, inv)
                          * (X - mu), axis=1))
        return constant * exponent

    def fit(self, X):
        self.initialize_parameters(X)

        for _ in range(self.n_iterations):
            # E-step
            probabilities = self.calculate_probabilities(X)
            responsibilities = probabilities / \
                np.sum(probabilities, axis=1)[:, np.newaxis]

            # M-step
            N_k = np.sum(responsibilities, axis=0)
            self.mu = np.dot(responsibilities.T, X) / N_k[:, np.newaxis]
            self.sigma = [np.dot(responsibilities[:, i] * (X - self.mu[i]).T, (X - self.mu[i])) / N_k[i]
                          for i in range(self.n_components)]
            self.weights = N_k / X.shape[0]

    def predict(self, X):
        probabilities = self.calculate_probabilities(X)
        return np.argmax(probabilities, axis=1)


if __name__ == "__main__":
    gmm = GMM_EM(n_components=2)
    gmm.fit(X)

    labels = gmm.predict(X)

    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.scatter(gmm.mu[:, 0], gmm.mu[:, 1], color='red', marker='x', s=100)
    plt.title('GMM Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

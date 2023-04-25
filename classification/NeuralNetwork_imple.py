import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler


def activate_forward(z):
    return np.tanh(z)


def activate_backward(a):
    return 1.0 - np.tanh(a)**2


def out_forward(z):
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z)


class Net:
    def __init__(self, sizes):
        self.n_layers = len(sizes)
        self.sizes = sizes
        self.w = [np.random.randn(x, y) for x, y in zip(sizes[:-1], sizes[1:])]
        self.b = [np.random.randn(1, y) for y in sizes[1:]]
        # self.b1 = np.zeros((1, hidden_dim))

    def fit(self, x, y):
        grad_b = [np.zeros(b.shape) for b in self.b]
        grad_w = [np.zeros(w.shape) for w in self.w]
        # forward
        a = x.reshape(1, -1)
        a_list = [a]
        z_list = []
        for w, b in zip(self.w[:-1], self.b[:-1]):
            z = np.dot(a, w) + b
            z_list.append(z)
            a = activate_forward(z)
            a_list.append(a)
        z_out = np.dot(a, self.w[-1]) + self.b[-1]
        a_out = out_forward(z_out)
        # backward
        delta = a_out-y
        grad_w[-1] = np.dot(np.array(a_list[-1]).T, delta)
        grad_b[-1] = delta
        for l in range(2, self.n_layers):
            delta = np.dot(delta, self.w[-l+1].T) * \
                activate_backward(a_list[-l+1])
            grad_w[-l] = np.dot(a_list[-l].T, delta)
            grad_b[-l] = delta
        return grad_w, grad_b

    def update(self, batch_data, eta):
        delta_w = [np.zeros(w.shape) for w in self.w]
        delta_b = [np.zeros(b.shape) for b in self.b]
        for x, y in batch_data:
            grad_w, grad_b = self.fit(x, y)
            delta_w = [dw+gw for dw, gw in zip(delta_w, grad_w)]
            delta_b = [db+gb for db, gb in zip(delta_b, grad_b)]
        self.w = [w-(eta/len(batch_data))*dw for w,
                  dw in zip(self.w, delta_w)]
        self.b = [b-(eta/len(batch_data))*db for b,
                  db in zip(self.b, delta_b)]

    def predict(self, a):
        for w, b in zip(self.w[:-1], self.b[:-1]):
            a = activate_forward(np.dot(a, w) + b)
        z_out = np.dot(a, self.w[-1]) + self.b[-1]
        a_out = out_forward(z_out)
        return a_out

    def evaluate(self, data):
        results = [(np.argmax(self.predict(x), axis=1),
                    np.argmax(y)) for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def train(self, train_data, epochs, mini_batch_size, eta, valid_data):
        for epoch in range(epochs):
            random.shuffle(train_data)
            batch_datas = [train_data[k:k+mini_batch_size]
                           for k in range(0, len(train_data), mini_batch_size)]
            for batch_data in batch_datas:
                self.update(batch_data, eta)
            print("Epoch {}: Train {}, Valid {}".format(epoch,
                                                        self.evaluate(
                                                            train_data)/len(train_data),
                                                        self.evaluate(valid_data)/len(valid_data)))


if __name__ == "__main__":
    import os
    from pathlib import Path

    parent_path = Path(__file__).parents[1]
    paths = ['data_sets', 'MNIST']
    train_data = pd.read_csv(os.path.join(parent_path, *paths, 'train.csv'))
    test_data = pd.read_csv(os.path.join(parent_path, *paths, 'test.csv'))
    X_train, X_valid, y_train, y_valid = map(np.array, train_test_split(
        train_data.drop(columns='label'), train_data['label'], test_size=0.2))
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    onehot = LabelBinarizer()
    Y_train = onehot.fit_transform(y_train)
    Y_valid = onehot.fit_transform(y_valid)

    model = Net([784, 256, 64, 10])
    model.train(train_data=list(zip(X_train, Y_train)),
                valid_data=list(zip(X_valid, Y_valid)),
                eta=1e-3, epochs=50, mini_batch_size=10)

    df_res = pd.DataFrame(
        data=np.argmax(model.predict(test_data), axis=1), columns=['label'])
    df_res.index.name = 'id'
    df_res.to_csv('res.csv')

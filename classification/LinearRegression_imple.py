import os
from math import ceil, floor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def train_test_split(X, Y, train_size):
    train_n = train_size*np.shape(X)[0]
    X_train = X[:ceil(train_n)]
    Y_train = Y[:ceil(train_n)]
    X_test = X[floor(train_n):]
    Y_test = Y[floor(train_n):]
    return X_train, X_test, Y_train, Y_test


def preprocess(input):
    offset = np.average(input, axis=0)
    offset.astype(input.dtype, copy=True)
    if input.ndim == 1:
        scale = 1.0
    else:
        scale = np.ones(input.shape[1], dtype=input.dtype)
    # scale = np.std(input, axis=0)
    # scale.astype(input.dtype, copy=True)
    return (input-offset)/scale


def forecast(W, X):
    if W.ndim == 1:
        W = np.matrix(W)
    z = np.dot(X, W.T)
    z = np.array(z, dtype=np.float64)
    fc = 1.0/(1.0+np.exp(-z))
    # fc = np.frompyfunc(lambda x: 1e-5 if x < 0 else x, 1, 1)(z)
    return fc


def fit(X, Y):
    X_matrix = np.matrix(X)
    Y_matrix = np.matrix(Y)
    if Y.ndim == 1:
        Y_matrix = Y_matrix.T
    W = np.array((X_matrix.T*X_matrix).I*X_matrix.T*Y_matrix)
    if Y.ndim == 1:
        W = np.ravel(W)
    return W


def loss(Y, Y_forecast):
    loss_sum = 0.0
    m = np.shape(Y)[0]
    for i in range(m):
        loss_sum -= (np.log(Y_forecast[i, 0])*Y[i, 0]) + \
            (np.log(1-Y_forecast[i, 0])*(1-Y[i, 0]))
    return loss_sum / m


if __name__ == "__main__":
    parent_path = Path(__file__).parents[1]
    paths = ['data_sets', 'diabetes']
    train_data = pd.read_csv(os.path.join(parent_path, *paths, 'train.csv'))
    test_data = pd.read_csv(os.path.join(parent_path, *paths, 'test.csv'))
    train_data = train_data.sample(frac=1)
    X_train,  X_test, Y_train, Y_test = train_test_split(train_data.drop(
        columns='diabetes'), train_data['diabetes'], train_size=0.8)
    X_train = preprocess(np.array(X_train))
    X_test = preprocess(np.array(X_test))
    X_target = preprocess(np.array(test_data))
    Y_train = preprocess(np.array(Y_train))
    Y_test = preprocess(np.array(Y_test))

    # des = train_data.describe()
    # plt.xlabel('glucose')
    # plt.ylabel('bloodpressure')
    # plt.xlim(xmax=des.loc['max', 'glucose'], xmin=des.loc['min', 'glucose'])
    # plt.ylim(ymax=des.loc['max', 'bloodpressure'],
    #          ymin=des.loc['min', 'bloodpressure'])
    # for id, series in train_data.iterrows():
    #     color = ('#00CED1'if series['diabetes']else '#DC143C')
    #     plt.scatter(series['glucose'],
    #                 series['bloodpressure'], c=color)
    # plt.show()

    X_train = np.hstack((X_train, np.ones((np.shape(X_train)[0], 1))))
    W = fit(X_train, Y_train)

    X_test = np.hstack((X_test, np.ones((np.shape(X_test)[0], 1))))
    print("\n验证损失:"+str(loss(np.mat(Y_test).reshape(-1, 1), forecast(W, X_test))))

    X_target = np.hstack((X_target, np.ones((np.shape(X_target)[0], 1))))
    df_res = pd.DataFrame(data=forecast(W, X_target), columns=['diabetes'])
    df_res['diabetes'] = df_res['diabetes'].apply(
        lambda x: 1 if x > 0.5 else 0)
    df_res.index.name = 'id'
    df_res.to_csv('res.csv')

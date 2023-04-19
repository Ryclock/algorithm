import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    parent_path = Path(__file__).parents[1]
    paths = ['data_sets', 'diabetes']
    train_data = pd.read_csv(os.path.join(parent_path, *paths, 'train.csv'))
    test_data = pd.read_csv(os.path.join(parent_path, *paths, 'test.csv'))
    # print(train_data.describe())
    # print(train_data.info())
    # print(test_data.describe())
    # print(test_data.info())

    X_train, X_valid, Y_train, Y_valid = train_test_split(
        train_data.drop(columns='diabetes'), train_data['diabetes'], train_size=0.8)

    # X_train = train_data.drop(columns='diabetes')
    # Y_train = train_data['diabetes']
    # print(train_data.columns)
    # n_rows = 1
    # n_columns = 2
    # plt.figure(figsize=(n_columns*6, n_rows*5))
    # for i, col in enumerate(train_data.columns):
    #     if col == "diabetes":
    #         continue
    #     X_train = train_data.loc[:, col].values.reshape(-1, 1)
    #     Y_train = train_data['diabetes']
    #     model = LinearRegression()
    #     model.fit(X_train, Y_train)
    #     score = model.score(X_train, Y_train)
    #     axes = plt.subplot(n_rows, n_columns, i+1)
    #     plt.scatter(X_train, Y_train, c='blue')
    #     x = np.linspace(X_train.min(), X_train.max(), 100)
    #     y = model.coef_ * x + model.intercept_
    #     plt.plot(x, y, c='red')
    #     axes.set_title(col + ':' + str(score))
    # plt.show()

    model = LinearRegression()
    model.fit(X_train, Y_train)
    score = model.score(X_valid, Y_valid)
    print(model.coef_, model.intercept_)
    print(score)

    df_res = pd.DataFrame(data=model.predict(test_data), columns=['diabetes'])
    df_res['diabetes'] = df_res['diabetes'].apply(
        lambda x: 1 if x > 0.5 else 0)
    df_res.index.name = 'id'
    # df_res.to_csv('res.csv')

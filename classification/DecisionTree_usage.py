from sklearn.tree import DecisionTreeClassifier
from pathlib import Path
import os
import pandas as pd

if __name__ == "__main__":
    parent_path = Path(__file__).parents[1]
    paths = ['data_sets', 'watermelon']
    df = pd.read_csv(filepath_or_buffer=os.path.join(
        parent_path, *paths, 'data.csv'), index_col=0)
    X_test = [['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑']]
    for column in df.columns:
        value_list = df[column].value_counts().keys().to_list()
        df[column] = df[column].apply(lambda x: value_list.index(x))
        i = df.columns.tolist().index(column)
        if i < len(df.columns)-1:
            X_test[0][i] = value_list.index(X_test[0][i])
    X_train = df.iloc[:, :-1].values
    Y_train = df.iloc[:, -1].values
    model = DecisionTreeClassifier(criterion='entropy')
    model.fit(X_train, Y_train)
    score = model.score(X_train, Y_train)
    print(score)
    res = model.predict(X_test)
    print(res)

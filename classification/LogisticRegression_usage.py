from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    data = load_breast_cancer()
    X = data['data']
    y = data['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    X_train_norm = minmax_scale(X_train)
    X_test_norm = minmax_scale(X_test)

    module = LogisticRegression(max_iter=10000)
    module.fit(X_train_norm, y_train)
    y_pred = module.predict(X_test_norm)
    acc = accuracy_score(y_test, y_pred)
    print(acc)

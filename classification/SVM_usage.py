from sklearn import svm
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    data = load_breast_cancer()
    X = data['data']
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    model = svm.SVC(C=1, kernel='linear')
    model.fit(X_train_sc, y_train)

    y_pred = model.predict(X_test_sc)
    y_pred[y_pred > 0] = 1
    y_pred[y_pred <= 0] = 0

    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)

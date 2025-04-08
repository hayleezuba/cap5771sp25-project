from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pandas as pd


def load_data():
    X_train = pd.read_csv('X_train.csv')
    X_test = pd.read_csv('X_test.csv')
    y_train = pd.read_csv('y_train.csv')
    y_test = pd.read_csv('y_test.csv')
    return X_train, X_test, y_train, y_test


def train_and_evaluate():
    X_train, X_test, y_train, y_test = load_data()

    # Handle shape errors
    if y_train.shape[1] > 1:
        y_train = y_train.idxmax(axis=1)
        y_test = y_test.idxmax(axis=1)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train.values.ravel())
    rf_preds = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    print(f"Random Forest Accuracy: {rf_acc:.4f}")

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, solver='lbfgs')
    lr.fit(X_train, y_train.values.ravel())
    lr_preds = lr.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_preds)
    print(f"Logistic Regression Accuracy: {lr_acc:.4f}")

    # Neural Network
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
    mlp.fit(X_train, y_train.values.ravel())
    mlp_preds = mlp.predict(X_test)
    mlp_acc = accuracy_score(y_test, mlp_preds)
    print(f"Neural Network Accuracy: {mlp_acc:.4f}")


def main():
    train_and_evaluate()


if __name__ == '__main__':
    main()

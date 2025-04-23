from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from joblib import dump, load
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
    dump(rf, '../trained_models/rf.pkl')
    print(f"Random Forest Accuracy: {rf_acc:.4f}")

    # Neural Network
    mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=200, random_state=42)
    mlp.fit(X_train, y_train.values.ravel())
    mlp_preds = mlp.predict(X_test)
    mlp_acc = accuracy_score(y_test, mlp_preds)
    dump(mlp, '../trained_models/neural_network.pkl')
    print(f"Neural Network Accuracy: {mlp_acc:.4f}")

    # Linear SVM (replacement for Logistic Regression)
    svm = LinearSVC(dual=False, C=1.0, max_iter=1000, tol=1e-4, random_state=42)
    svm.fit(X_train, y_train.values.ravel())
    svm_preds = svm.predict(X_test)
    svm_acc = accuracy_score(y_test, svm_preds)
    dump(svm, '../trained_models/svm.pkl')
    print(f"Linear SVM Accuracy: {10 * svm_acc:.4f}")


def main():
    train_and_evaluate()


if __name__ == '__main__':
    main()

from sklearn.metrics import f1_score, roc_auc_score, precision_score
from joblib import load
import pandas as pd


def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)
    print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")


def main():

    # Load test data
    X_test = pd.read_csv("feature engineering/X_test.csv")
    y_test = pd.read_csv("feature engineering/y_test.csv")

    rf_model = load("trained_models/rf.pkl")
    SVM_model = load("trained_models/svm.pkl")
    NN_model = load("trained_models/neural_network.pkl")
    print(f"Evaluating model: Random Forest\n----------")
    evaluate_model(rf_model, X_test, y_test)
    print(f"Evaluating model: SVM\n----------")
    evaluate_model(SVM_model, X_test, y_test)
    print(f"Evaluating model: Neural Network\n----------")
    evaluate_model(NN_model, X_test, y_test)


if __name__ == "__main__":
    main()
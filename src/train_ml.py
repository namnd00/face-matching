import random
import pandas as pd
import os
import numpy as np
from keras.utils import to_categorical

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score


def train_by_logistic_regression(X_train, X_test, y_train, y_test):
    print('[INFO] Logistic Regression...')
    lr = LogisticRegression(solver='lbfgs')
    lr.fit(X_train, y_train)
    train_score_lr = round(accuracy_score(y_train, lr.predict(X_train)) * 100, 2)
    test_score_lr = round(accuracy_score(y_test, lr.predict(X_test)) * 100, 2)
    print("+ Train Result: ", train_score_lr)
    print("+ Test Result: ", test_score_lr)

    return train_score_lr, test_score_lr


def train_by_random_forest(X_train, X_test, y_train, y_test):
    print('[INFO] Random Forest Classifier...')
    train_score_rfc = []
    test_score_rfc = []
    rfc = RandomForestClassifier(n_estimators=1000)
    rfc.fit(X_train, y_train)
    train_score_rfc.append(round(accuracy_score(y_train, rfc.predict(X_train)) * 100, 2))
    test_score_rfc.append(round(accuracy_score(y_test, rfc.predict(X_test)) * 100, 2))
    print("+ Train Result: ", train_score_rfc)
    print("+ Test Result: ", test_score_rfc)
    max_value = max(test_score_rfc)
    max_index = test_score_rfc.index(max_value)
    train_score_rfc = train_score_rfc[max_index]
    test_score_rfc = test_score_rfc[max_index]

    return train_score_rfc, test_score_rfc


def train_by_svm(X_train, X_test, y_train, y_test):
    print('[INFO] SVM...')
    svm = SVC(kernel='rbf', gamma=0.1, C=1.0)
    train_score_svm = round(accuracy_score(y_train, svm.predict(X_train)) * 100, 2)
    test_score_svm = round(accuracy_score(y_test, svm.predict(X_test)) * 100, 2)
    print("+ Train Result: ", train_score_svm)
    print("+ Test Result: ", test_score_svm)

    return train_score_svm, test_score_svm


def get_y_true(df):
    y_true = df.label.values.tolist()
    return np.array(y_true)


def train(Xtrain, Xtest, ytrain, ytest, output, name_df):
    train_score_lr, test_score_lr = train_by_logistic_regression(Xtrain, Xtest, ytrain, ytest)
    train_score_rfc, test_score_rfc = train_by_random_forest(Xtrain, Xtest, ytrain, ytest)
    train_score_svm, test_score_svm = train_by_svm(Xtrain, Xtest, ytrain, ytest)

    models_dict = {
        'Train Accuracy': [train_score_lr, train_score_rfc, train_score_svm],
        'Test Accuracy': [test_score_lr, test_score_rfc, test_score_svm]
    }

    models_df = pd.DataFrame(models_dict,
                             index=['Logistic Regression', 'Random Forest Classifier', 'Support Vector Machine'])

    output_path = os.path.join(output, name_df)
    models_df.to_csv(output_path)


if __name__ == "__main__":
    path = "../models_src"

    output = "results_ml"
    if not os.path.exists(output):
        os.mkdir(output)

    for model_dir in os.listdir(path):
        if ".idea" in model_dir:
            continue
        model_path = os.path.join(path, model_dir)

        test_path = os.path.join(model_path, "sample_submission.csv")
        train_path = os.path.join(model_path, "train.csv")

        train_data = os.path.join(model_path, "train_data.npy")
        test_data = os.path.join(model_path, "test_data.npy")

        test_df = pd.read_csv(test_path)
        train_df = pd.read_csv(train_path)

        xTrain = np.load(train_data)
        yTrain = get_y_true(train_df)

        xTest = np.load(test_data)
        yTest = get_y_true(test_df)

        nsamples, nx, ny = xTrain.shape
        xTrain = xTrain.reshape((nsamples, nx * ny))

        train(xTrain, xTest, yTrain, yTest, output, model_dir)

import random
import pandas as pd
import os
import numpy as np
from keras.utils import to_categorical

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score


NUMBER_OF_PARTS = 5
NUMBER_OF_FOLDS = 5
NUMBER_OF_CLASSES = 2


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
    for i in range(1, 1000, 100):
        rfc = RandomForestClassifier(n_estimators=i)
        rfc.fit(X_train, y_train)
        train_score_rfc.append(round(accuracy_score(y_train, rfc.predict(X_train)) * 100, 2))
        test_score_rfc.append(round(accuracy_score(y_test, rfc.predict(X_test)) * 100, 2))
        print(i)
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
    test_df = pd.read_csv('../../datasets/test_refined.csv')
    train_df = pd.read_csv('../../datasets/train_refined.csv')

    xtrain = np.load('train_data.npy')
    xtrain_aug = np.load('train_aug_data.npy')
    ytrain = get_y_true(train_df)
    xtest = np.load('test_data.npy')

    output = "results_ml"
    if not os.path.exists(output):
        os.mkdir(output)

    for part in random.sample(range(10), NUMBER_OF_PARTS):
        for fold in range(NUMBER_OF_FOLDS):
            v_df = train_df.loc[train_df['rt%d' % part] == fold]
            vidxs = v_df.index.values.tolist()
            t_df = train_df.loc[~train_df.index.isin(v_df.index)]
            tidxs = t_df.index.values.tolist()
            print('**************Part %d    Fold %d**************' % (part, fold))

            xtrain_fold = xtrain_aug[tidxs, :, :]
            ytrain_fold = ytrain[tidxs, :]

            xvalid_fold = xtrain[vidxs, :]
            yvalid_fold = ytrain[vidxs, :]

            train_size = len(tidxs)
            valid_size = len(vidxs)

            nsamples, nx, ny = xtrain_fold.shape
            xtrain_fold = xtrain_fold.reshape((nsamples, nx * ny))

            name_df = f"part{part}.fold{fold}.csv"
            print('TRAIN SIZE: %d VALID SIZE: %d' % (train_size, valid_size))

            train(xtrain_fold, xvalid_fold, ytrain_fold, yvalid_fold, output, name_df)

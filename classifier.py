"""A series of classifier.

Including:
    RandomForest,
    AdaBoost,
    Logistic,
    SVM,
    KNN,
    GaussianNB
"""
import numpy as np
import pandas as pd
from time import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from tools import get_train_test


def random_forest_classifier(df, features_list=None, weighted=False):
    """Return best random forest classifier by GridSearchCV.

    Args:
        df: dataset in dataframe format.
        features_list: features used in training classifier.
        weighted: True will use f1_weighted score to evaluate classifier
            for catch more True value in imbalance label,
            otherwise use f1 score.
    """
    if features_list is None:
        features_list = [i for i in df.drop('poi', axis=1)]

    features_train, features_test, labels_train, labels_test = get_train_test(
        df, features_list=features_list)

    if weighted:
        scoring = "f1_weighted"
    else:
        scoring = "f1"

    param_grid = {
        'n_estimators': list(range(50, 201, 25)),
        'max_depth': list(range(3, 7))

    }

    clf = RandomForestClassifier(random_state=42, class_weight="balanced",
                                 oob_score=True, min_samples_leaf=2,
                                 n_jobs=-1)

    clf = GridSearchCV(clf, param_grid, scoring=scoring, verbose=0)

    print("Fitting the classifier to the training set")
    t0 = time()
    clf.fit(features_train, labels_train)
    print("done in %0.3fs" % (time() - t0))
    print(clf.best_estimator_)
    clf = clf.best_estimator_

    # Quantitative evaluation of the model quality on the training set
    print("Predicting on the training set")
    t0 = time()
    train_pred = clf.predict(features_train)
    print(("Done in %0.3fs" % (time() - t0)))

    print((classification_report(labels_train, train_pred)))
    print((confusion_matrix(labels_train, train_pred)))

    # Quantitative evaluation of the model quality on the test set
    print("Predicting on the testing set")
    t0 = time()
    pred = clf.predict(features_test)
    print(("Done in %0.3fs" % (time() - t0)))

    print((classification_report(labels_test, pred)))
    print((confusion_matrix(labels_test, pred)))

    sort_index = (-clf.feature_importances_).argsort()
    print(clf.feature_importances_[sort_index])
    feature_sort_by_importance = np.asarray(features_list)[sort_index]
    print(feature_sort_by_importance)

    return clf


def adaboost_classifier(df, features_list=None, weighted=False):
    """Return best adaboost classifier by GridSearchCV.

    Args:
        df: dataset in dataframe format.
        features_list: features used in training classifier.
        weighted: True will use f1_weighted score to evaluate classifier
            for catch more True value in imbalance label,
            otherwise use f1 score.
    """
    if features_list is None:
        features_list = [i for i in df.drop('poi', axis=1)]

    features_train, features_test, labels_train, labels_test = get_train_test(
        df, features_list=features_list)

    if weighted:
        scoring = "f1_weighted"
    else:
        scoring = "f1"

    clf = AdaBoostClassifier(random_state=42)

#     kbest = SelectKBest(f_classif)
#     pipe = Pipeline([('kbest', kbest), ('adaboost', clf)])

    param_grid = {
        'n_estimators': list(range(50, 201, 25)),
        'learning_rate': [0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5]
        #         'kbest__k': list(range(5, 9))
    }

    clf = GridSearchCV(clf, param_grid, scoring=scoring, verbose=0)
#     clf = GridSearchCV(clf, param_grid, scoring='recall')

    print("Fitting the classifier to the training set")
    t0 = time()
    clf.fit(features_train, labels_train)
    print("done in %0.3fs" % (time() - t0))

    print(clf.best_estimator_)
    clf = clf.best_estimator_

    # Quantitative evaluation of the model quality on the training set
    print("Predicting on the training set")
    t0 = time()
    train_pred = clf.predict(features_train)
    print(("Done in %0.3fs" % (time() - t0)))

    print((classification_report(labels_train, train_pred)))
    print((confusion_matrix(labels_train, train_pred)))

    # Quantitative evaluation of the model quality on the test set
    print("Predicting on the testing set")
    t0 = time()
    pred = clf.predict(features_test)
    print(("Done in %0.3fs" % (time() - t0)))

    print((classification_report(labels_test, pred)))
    print((confusion_matrix(labels_test, pred)))

    sort_index = (-clf.feature_importances_).argsort()
    print(clf.feature_importances_[sort_index])
    feature_sort_by_importance = np.asarray(features_list)[sort_index]
    print(feature_sort_by_importance)

    return clf


def logistic_regression_classifier(df, features_list=None, weighted=False):
    """Return best logistic classifier by GridSearchCV.

    Args:
        df: dataset in dataframe format.
        features_list: features used in training classifier.
        weighted: True will use f1_weighted score to evaluate classifier
            for catch more True value in imbalance label,
            otherwise use f1 score.
    """
    features_train, features_test, labels_train, labels_test = get_train_test(
        df, features_list=features_list)

    if weighted:
        scoring = "f1_weighted"
    else:
        scoring = "f1"

    clf = LogisticRegression(class_weight='balanced')

    t0 = time()
    param_grid = {
        'C': [0.05, 0.1, 0.5, 1, 5, 10],
    }

    clf = GridSearchCV(clf, param_grid, scoring=scoring, verbose=0)
#     clf = GridSearchCV(clf, param_grid, scoring='recall')
    print("Fitting the classifier to the training set")
    t0 = time()
    clf.fit(features_train, labels_train)
    print("done in %0.3fs" % (time() - t0))

    print(clf.best_estimator_)

    # Quantitative evaluation of the model quality on the training set
    print("Predicting on the training set")
    t0 = time()
    train_pred = clf.predict(features_train)
    print(("Done in %0.3fs" % (time() - t0)))

    print((classification_report(labels_train, train_pred)))
    print((confusion_matrix(labels_train, train_pred)))

    # Quantitative evaluation of the model quality on the test set
    print("Predicting on the testing set")
    t0 = time()
    pred = clf.predict(features_test)
    print(("Done in %0.3fs" % (time() - t0)))

    print((classification_report(labels_test, pred)))
    print((confusion_matrix(labels_test, pred)))

    return clf


def nb_classifier(df, features_list=None, weighted=False):
    """Return GaussianNB classifier.

    Args:
        df: dataset in dataframe format.
        features_list: features used in training classifier.
        weighted: True will use f1_weighted score to evaluate classifier
            for catch more True value in imbalance label,
            otherwise use f1 score.
    """
    features_train, features_test, labels_train, labels_test = get_train_test(
        df, features_list=features_list)

    if weighted:
        scoring = "f1_weighted"
    else:
        scoring = "f1"

    print("Fitting the classifier to the training set")
    t0 = time()

    clf = GaussianNB()
#     clf = GridSearchCV(clf, param_grid, scoring='recall')
    clf = clf.fit(features_train, labels_train)
    print("done in %0.3fs" % (time() - t0))

    # Quantitative evaluation of the model quality on the training set
    print("Predicting on the training set")
    t0 = time()
    train_pred = clf.predict(features_train)
    print(("Done in %0.3fs" % (time() - t0)))

    print((classification_report(labels_train, train_pred)))
    print((confusion_matrix(labels_train, train_pred)))

    # Quantitative evaluation of the model quality on the test set
    print("Predicting on the testing set")
    t0 = time()
    pred = clf.predict(features_test)
    print(("Done in %0.3fs" % (time() - t0)))

    print((classification_report(labels_test, pred)))
    print((confusion_matrix(labels_test, pred)))
    return clf


def knn_classifier(df, features_list=None, weighted=False):
    """Return best k nearest neighbors classifier by GridSearchCV.

    Args:
        df: dataset in dataframe format.
        features_list: features used in training classifier.
        weighted: True will use f1_weighted score to evaluate classifier
            for catch more True value in imbalance label,
            otherwise use f1 score.
    """
    features_train, features_test, labels_train, labels_test = get_train_test(
        df, features_list=features_list)

    if weighted:
        scoring = "f1_weighted"
    else:
        scoring = "f1"

    print("Fitting the classifier to the training set")
    t0 = time()
    param_grid = {
        'n_neighbors': list(range(5, 20, 2)),
    }

    clf = KNeighborsClassifier()
    clf = GridSearchCV(clf, param_grid, scoring=scoring, verbose=0, n_jobs=-1)
#     clf = GridSearchCV(clf, param_grid, scoring='recall')
    clf = clf.fit(features_train, labels_train)
    print("done in %0.3fs" % (time() - t0))

    print("Best estimator found by grid search:")
    print(clf.best_estimator_)

    # Quantitative evaluation of the model quality on the training set
    print("Predicting on the training set")
    t0 = time()
    train_pred = clf.predict(features_train)
    print(("Done in %0.3fs" % (time() - t0)))

    print((classification_report(labels_train, train_pred)))
    print((confusion_matrix(labels_train, train_pred)))

    # Quantitative evaluation of the model quality on the test set
    print("Predicting on the testing set")
    t0 = time()
    pred = clf.predict(features_test)
    print(("Done in %0.3fs" % (time() - t0)))

    print((classification_report(labels_test, pred)))
    print((confusion_matrix(labels_test, pred)))
    return clf


def svm_classifier(df, features_list=None, weighted=False):
    """Return best svm classifier by GridSearchCV.

    Args:
        df: dataset in dataframe format.
        features_list: features used in training classifier.
        weighted: True will use f1_weighted score to evaluate classifier
            for catch more True value in imbalance label,
            otherwise use f1 score.
    """
    features_train, features_test, labels_train, labels_test = get_train_test(
        df, features_list=features_list)

    if weighted:
        scoring = "f1_weighted"
    else:
        scoring = "f1"

    print("Fitting the classifier to the training set")
    t0 = time()
    param_grid = {
        'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10],
        'gamma': [1e-3, 5e-3, 0.01, 0.05, 0.1],
    }

    clf = SVC(kernel='rbf', class_weight="balanced")
    clf = GridSearchCV(clf, param_grid, scoring=scoring, verbose=0)
#     clf = GridSearchCV(clf, param_grid, scoring='recall')
    clf = clf.fit(features_train, labels_train)
    print("done in %0.3fs" % (time() - t0))

    print("Best estimator found by grid search:")
    print(clf.best_estimator_)

    # Quantitative evaluation of the model quality on the training set
    print("Predicting on the training set")
    t0 = time()
    train_pred = clf.predict(features_train)
    print(("Done in %0.3fs" % (time() - t0)))

    print((classification_report(labels_train, train_pred)))
    print((confusion_matrix(labels_train, train_pred)))

    # Quantitative evaluation of the model quality on the test set
    print("Predicting on the testing set")
    t0 = time()
    pred = clf.predict(features_test)
    print(("Done in %0.3fs" % (time() - t0)))

    print((classification_report(labels_test, pred)))
    print((confusion_matrix(labels_test, pred)))
    return clf

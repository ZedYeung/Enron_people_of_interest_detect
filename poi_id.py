"""This project is aimed to identified the person of interest in Enron scandal.

In 2000, Enron was one of the largest companies in the United States.
By 2002, it had collapsed into bankruptcy due to widespread corporate fraud.
In the resulting Federal investigation, a significant amount of typically
confidential information entered into the public record, including tens of
thousands of emails and detailed financial data for top executives.
In this project, I will try to build a detecitve model to identify the
person of interest in Enron scandal, based on financial and email data
made public as a result of the Enron scandal.
"""
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from time import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

np.random.seed(401)


def minmax_df(df):
    """Scale the features to 0-1 range by min-max scaling."""
    minmax_features = []
    for i in df:
        if i not in ["poi", "raction_from_poi_to_this_person",
                     "fraction_from_this_person_to_poi",
                     "fraction_shared_receipt_with_poi"]:
            minmax_features.append(i)

    minmax_scaler = MinMaxScaler()
    minmax_scale_df = df.copy()
    minmax_scale_df[minmax_features] = minmax_scaler.fit_transform(
        minmax_scale_df[minmax_features])

    return minmax_scale_df


def get_train_test(df, features_list=None):
    """Get train and test dataset of features and labels.

    Args:
        df: dataset in dataframe format.
        features_list: features used in training classifier.
    """
    labels = df['poi']
    if features_list is None:
        features = df.drop('poi', axis=1)
    else:
        if "poi" in features_list:
            features_list.remove("poi")
        features = df[features_list]
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=42)
    return features_train, features_test, labels_train, labels_test


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


# In[9]:

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


def fraction_feature(part, total):
    """Scale feature by percentile.

    Args:
        part: feature need to scale.
        total: the sum related with the feature.
    """
    fraction_feature_list = []
    for i in data_dict:
        if data_dict[i][part] == "NaN" or data_dict[i][total] == "NaN":
            fraction = 0
        else:
            fraction = data_dict[i][part] / data_dict[i][total]
        data_dict[i]["fraction_" + part] = fraction
    return data_dict


def univariate_plot(dataframe):
    """Plot all features' distribution bar chart."""
    df = dataframe.copy()
    df['color'] = df.apply(lambda row: "red" if row["poi"] else "blue", axis=1)
    plot_columns = [i for i in df if i not in ["color", "poi"]]
    for column in plot_columns:
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.set_yticklabels([])
        ax.set_title(column, fontsize=30)
        plt.bar(range(0, len(df)), df[column],  color=df["color"])


def correlation_matrix(df, features_list):
    """Output features correlation matrix and write to html.

    Output features correlation matrix that styling by color shade.

    Args:
        df: dataset in dataframe format.
        features_list: features that will show in correlation matrix.
    """
    if "poi" in features_list:
        features_list.remove("poi")
    features_df = df[features_list]

    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    corr = features_df.corr()

    def magnify():
        return [dict(selector="th",
                     props=[("font-size", "7pt")]),
                dict(selector="td",
                     props=[('padding', "0em 0em")]),
                dict(selector="th:hover",
                     props=[("font-size", "12pt")]),
                dict(selector="tr:hover td:hover",
                     props=[('max-width', '200px'),
                            ('font-size', '12pt')])
                ]

    corr_style = corr.style.background_gradient(cmap, axis=1)\
        .set_properties(**{'max-width': '80px', 'font-size': '10pt'})\
        .set_caption("Features Correlation Matrix")\
        .set_precision(2)\
        .set_table_styles(magnify())
    corr_matrix_html = corr_style.render()

    with open(features_list[1] + '.html', 'w') as f:
        f.write(corr_matrix_html)

    return corr_style


def classifier_test(df, features_list=None, weighted=True):
    """Train and test on all classifier."""
    for classifier in ["random_forest_classifier", "adaboost_classifier",
                       "logistic_regression_classifier", "nb_classifier",
                       "knn_classifier", "svm_classifier"]:
        globals().get(classifier)(df, features_list, weighted)


def classifier_shuffle_split_test(df, features_list,
                                  classifiers, weighted=True):
    """Test series of classifier using StratifiedShuffleSplit."""
    for classifier in classifiers:
        clf = globals().get(classifier)(df, features_list, weighted)
        my_dataset = df.to_dict(orient="index")
        if "poi" not in features_list:
            features_list = ["poi"] + features_list
        test_classifier(clf, my_dataset, features_list)


# Task 1: Select what features you'll use.
# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".
# You will need to use more features
finance_features_list = ['poi', 'salary', 'deferral_payments',
                         'total_payments', 'loan_advances', 'bonus',
                         'restricted_stock_deferred',
                         'deferred_income', 'total_stock_value', 'expenses',
                         'exercised_stock_options', 'other',
                         'long_term_incentive', 'restricted_stock',
                         'director_fees']

email_features_list = ['poi', 'to_messages',
                       'from_poi_to_this_person', 'from_messages',
                       'from_this_person_to_poi', 'shared_receipt_with_poi']


# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

# Task 2: Remove outliers
data_dict.pop("TOTAL", 0)

# Task 3: Create new feature(s)
# Store to my_dataset for easy export below.
data_dict = fraction_feature("from_poi_to_this_person", "to_messages")
data_dict = fraction_feature("from_this_person_to_poi", "from_messages")
data_dict = fraction_feature("shared_receipt_with_poi", "to_messages")
df = pd.DataFrame.from_dict(data_dict, orient="index")

# Since I already transformed these email features into percentile,
# I will just discard these email features.
df = df.drop(["from_poi_to_this_person", "from_this_person_to_poi",
              "shared_receipt_with_poi", "email_address"], axis=1)

email_features_list = ['to_messages', 'from_messages',
                       'fraction_from_poi_to_this_person',
                       'fraction_from_this_person_to_poi',
                       'fraction_shared_receipt_with_poi']

features = finance_features_list[1:] + email_features_list

# check features NaN percentile
nan_percentile_dict = {}
for j in features:
    count = 0
    total_count = 0
    for i in data_dict:
        if data_dict[i][j] == "NaN":
            count += 1
        total_count += 1
    nan_percentile = count / total_count
    nan_percentile_dict[j] = nan_percentile
    print("%s nan percentile: %s" % (j, nan_percentile))

# fill the NaN with 0 for further minmax scaling.
df = df.replace('NaN', np.NaN)
df = df.fillna(0)

df_minmax = minmax_df(df)

# Just remove the features above 70% nan percent
df_minmax_nan_percent_filter = df_minmax.copy()
for i in nan_percentile_dict:
    if nan_percentile_dict[i] > 0.7:
        features.remove(i)
        df_minmax_nan_percent_filter = df_minmax_nan_percent_filter.drop(
            i, axis=1)

# use visualization to filter features.
# univariate_plot(df)

# From the deferred_income bar chart I find that the deferred_income
# should be transform to positive values so as to keep align with other
# features.
df_minmax_nan_percent_filter["deferred_income"] = \
    df_minmax_nan_percent_filter["deferred_income"].abs()

# After looking through the bar chart, I decide to remove 'from_messages',
# 'to_messages', 'total_payments' since they are
# all sum, noisy and having outliers.
df_minmax_filter = df_minmax_nan_percent_filter.copy()

for i in ['from_messages', 'to_messages', 'total_payments']:
    features.remove(i)
    df_minmax_filter = df_minmax_filter.drop(i, axis=1)

# correlation_matrix(df, finance_features_list)
# correlation_matrix(df, email_features_list)

# From the correlation matrix we can see that "exercised_stock_options",
# "restricted_stock", "total_stock_value" are highly related. So I decide
# to use PCA to merge them into one new feature.
df_stock_pca = df_minmax_filter.copy()

pca = PCA(n_components=1)

df_stock_pca["stock_pca"] = pca.fit_transform(
    df_stock_pca[["exercised_stock_options",
                  "restricted_stock", "total_stock_value"]])

df_stock_pca = df_stock_pca.drop(["exercised_stock_options",
                                  "restricted_stock",
                                  "total_stock_value"], axis=1)

for i in ["exercised_stock_options", "restricted_stock", "total_stock_value"]:
    features.remove(i)
features.append("stock_pca")

# Task 4: Try a varity of classifiers
# Please name your classifier clf for easy export below.
# Note that if you want to do PCA or other multi-stage operations,
# you'll need to use Pipelines. For more info:
# http://scikit-learn.org/stable/modules/pipeline.html
classifier_test(df_stock_pca)

# I noticed that both long_term_incentive and deferred_income are appeared
# on the last three importances in both random forest and adaboost.
# Consequently and last but not least, I remove them and make final features.
selected_features = ['bonus', 'expenses', 'salary', 'deferred_income', 'other',
                     'long_term_incentive', 'fraction_from_poi_to_this_person',
                     'fraction_from_this_person_to_poi',
                     'fraction_shared_receipt_with_poi', 'stock_pca']

df_select_feature = df_stock_pca.copy()
for i in ['deferred_income', 'long_term_incentive']:
    selected_features.remove(i)
    df_select_feature = df_select_feature.drop(i, axis=1)

# Task 5: Tune your classifier to achieve better than .3 precision and recall
# using our testing script. Check the tester.py script in the final project
# folder for details on the evaluation method, especially the test_classifier
# function. Because of the small size of the dataset, the script uses
# stratified shuffle split cross validation. For more info:
# http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Uncomment this to run StratifiedShuffleSplit test on all classifier.
# It will take long time to run.
# classifier_shuffle_split_test(df_select_feature, features,
#                               ["random_forest_classifier",
#                                "adaboost_classifier",
#                                "logistic_regression_classifier",
#                                "nb_classifier",
#                                "knn_classifier", "svm_classifier"])


# Task 6: Dump your classifier, dataset, and features_list so anyone can
# check your results. You do not need to change anything below, but make sure
# that the version of poi_id.py that you submit can be run on its own and
# generates the necessary .pkl files for validating your results.
clf = random_forest_classifier(df_select_feature)
my_dataset = df_select_feature.to_dict(orient="index")
features_list = ["poi"] + selected_features

dump_classifier_and_data(clf, my_dataset, features_list)

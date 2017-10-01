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
from tester import dump_classifier_and_data
from sklearn.decomposition import PCA
from tester import test_classifier
from classifier import *
from tools import *


np.random.seed(401)


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

# Uncomment this to test all classifier
# classifier_test(df_stock_pca)

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

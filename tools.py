"""Some tools used in poi_id."""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from classifier import *


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

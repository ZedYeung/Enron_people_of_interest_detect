


# Project Overview
This project is aimed to identified the person of interest in Enron scandal.

In 2000, Enron was one of the largest companies in the United States.
By 2002, it had collapsed into bankruptcy due to widespread corporate fraud.
In the resulting Federal investigation, a significant amount of typically
confidential information entered into the public record, including tens of
thousands of emails and detailed financial data for top executives.
In this project, I will try to build a detecitve model to identify the
person of interest in Enron scandal, based on financial and email data
made public as a result of the Enron scandal.

# Preparation
First of all, just load the dictionary containing the dataset.

And take all features into list.
```python
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
```

What is more, define four classifier used in this project:
random forest, adaboost, logistic and svm.

# Remove outliers
```
data_dict.pop("TOTAL", 0)
```

    {'bonus': 97343619,
     'deferral_payments': 32083396,
     'deferred_income': -27992891,
     'director_fees': 1398517,
     'email_address': 'NaN',
     'exercised_stock_options': 311764000,
     'expenses': 5235198,
     'from_messages': 'NaN',
     'from_poi_to_this_person': 'NaN',
     'from_this_person_to_poi': 'NaN',
     'loan_advances': 83925000,
     'long_term_incentive': 48521928,
     'other': 42667589,
     'poi': False,
     'restricted_stock': 130322299,
     'restricted_stock_deferred': -7576788,
     'salary': 26704229,
     'shared_receipt_with_poi': 'NaN',
     'to_messages': 'NaN',
     'total_payments': 309886585,
     'total_stock_value': 434509511}



# Create new features

Transform "from_poi_to_this_person", "from_this_person_to_poi", "shared_receipt_with_poi"
to percentile.

'fraction_from_poi_to_this_person' = from_poi_to_this_person" / "to_messages"    
'fraction_from_this_person_to_poi' =  from_this_person_to_poi" / "from_messages"    
'fraction_shared_receipt_with_poi' = "shared_receipt_with_poi" / "to_messages"    

# Check features NaN percentile


    salary nan percentile: 0.35172413793103446
    deferral_payments nan percentile: 0.7379310344827587
    total_payments nan percentile: 0.14482758620689656
    loan_advances nan percentile: 0.9793103448275862
    bonus nan percentile: 0.4413793103448276
    restricted_stock_deferred nan percentile: 0.8827586206896552
    deferred_income nan percentile: 0.6689655172413793
    total_stock_value nan percentile: 0.13793103448275862
    expenses nan percentile: 0.35172413793103446
    exercised_stock_options nan percentile: 0.30344827586206896
    other nan percentile: 0.36551724137931035
    long_term_incentive nan percentile: 0.5517241379310345
    restricted_stock nan percentile: 0.2482758620689655
    director_fees nan percentile: 0.8896551724137931
    to_messages nan percentile: 0.4068965517241379
    from_messages nan percentile: 0.4068965517241379
    fraction_from_poi_to_this_person nan percentile: 0.0
    fraction_from_this_person_to_poi nan percentile: 0.0
    fraction_shared_receipt_with_poi nan percentile: 0.0



# Minmax scaling
first fill the NaN with 0 and then do minmax scaling on all features except
"poi", "raction_from_poi_to_this_person", "fraction_from_this_person_to_poi", "fraction_shared_receipt_with_poi".

# First Test
```
Fitting the classifier to the training set
done in 13.479s
RandomForestClassifier(bootstrap=True, class_weight='balanced',
            criterion='gini', max_depth=None, max_features='auto',
            max_leaf_nodes=None, min_impurity_decrease=0.0,
            min_impurity_split=None, min_samples_leaf=2,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=50, n_jobs=-1, oob_score=True, random_state=42,
            verbose=0, warm_start=False)
Predicting on the training set
Done in 0.112s
             precision    recall  f1-score   support

      False       1.00      0.99      0.99        87
       True       0.93      1.00      0.97        14

avg / total       0.99      0.99      0.99       101

[[86  1]
 [ 0 14]]
Predicting on the testing set
Done in 0.107s
             precision    recall  f1-score   support

      False       0.93      1.00      0.96        40
       True       1.00      0.25      0.40         4

avg / total       0.94      0.93      0.91        44

[[40  0]
 [ 3  1]]
RandomForestClassifier(bootstrap=True, class_weight='balanced',
            criterion='gini', max_depth=None, max_features='auto',
            max_leaf_nodes=None, min_impurity_decrease=0.0,
            min_impurity_split=None, min_samples_leaf=2,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=50, n_jobs=-1, oob_score=True, random_state=42,
            verbose=0, warm_start=False)
	Accuracy: 0.86507	Precision: 0.48667	Recall: 0.21900	F1: 0.30207	F2: 0.24607
	Total predictions: 15000	True positives:  438	False positives:  462	False negatives: 1562	True negatives: 12538

Fitting the classifier to the training set
done in 26.598s
AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
          learning_rate=0.1, n_estimators=175, random_state=42)
Predicting on the training set
Done in 0.031s
             precision    recall  f1-score   support

      False       1.00      1.00      1.00        87
       True       1.00      1.00      1.00        14

avg / total       1.00      1.00      1.00       101

[[87  0]
 [ 0 14]]
Predicting on the testing set
Done in 0.027s
             precision    recall  f1-score   support

      False       0.95      0.93      0.94        40
       True       0.40      0.50      0.44         4

avg / total       0.90      0.89      0.89        44

[[37  3]
 [ 2  2]]
AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
          learning_rate=0.1, n_estimators=175, random_state=42)
	Accuracy: 0.84933	Precision: 0.40972	Recall: 0.29500	F1: 0.34302	F2: 0.31250
	Total predictions: 15000	True positives:  590	False positives:  850	False negatives: 1410	True negatives: 12150

Fitting the classifier to the training set
done in 2.940s
LogisticRegression(C=1, class_weight='balanced', dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=100,
          multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
          solver='liblinear', tol=1e-05, verbose=0, warm_start=False)
Predicting on the training set
Done in 0.003s
             precision    recall  f1-score   support

      False       0.95      0.80      0.87        87
       True       0.37      0.71      0.49        14

avg / total       0.87      0.79      0.82       101

[[70 17]
 [ 4 10]]
Predicting on the testing set
Done in 0.000s
             precision    recall  f1-score   support

      False       0.97      0.78      0.86        40
       True       0.25      0.75      0.38         4

avg / total       0.90      0.77      0.82        44

[[31  9]
 [ 1  3]]
GridSearchCV(cv=None, error_score='raise',
       estimator=LogisticRegression(C=1.0, class_weight='balanced', dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=100,
          multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
          solver='liblinear', tol=0.0001, verbose=0, warm_start=False),
       fit_params={}, iid=True, n_jobs=1,
       param_grid={'C': [0.1, 0.5, 1, 5, 10, 50, 100], 'tol': [1e-05, 5e-05, 0.0001, 0.0005, 0.001, 0.005, 0.01], 'max_iter': [100, 150, 200, 250, 300]},
       pre_dispatch='2*n_jobs', refit=True, scoring='f1', verbose=0)
	Accuracy: 0.74160	Precision: 0.27635	Recall: 0.57950	F1: 0.37423	F2: 0.47523
	Total predictions: 15000	True positives: 1159	False positives: 3035	False negatives:  841	True negatives: 9965

Fitting the classifier to the training set
done in 0.518s
Best estimator found by grid search:
SVC(C=0.5, cache_size=200, class_weight='balanced', coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
Predicting on the training set
Done in 0.001s
             precision    recall  f1-score   support

      False       0.93      0.76      0.84        87
       True       0.30      0.64      0.41        14

avg / total       0.84      0.74      0.78       101

[[66 21]
 [ 5  9]]
Predicting on the testing set
Done in 0.000s
             precision    recall  f1-score   support

      False       1.00      0.72      0.84        40
       True       0.27      1.00      0.42         4

avg / total       0.93      0.75      0.80        44

[[29 11]
 [ 0  4]]
GridSearchCV(cv=None, error_score='raise',
       estimator=SVC(C=1.0, cache_size=200, class_weight='balanced', coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False),
       fit_params={}, iid=True, n_jobs=1,
       param_grid={'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50], 'gamma': [0.001, 0.005, 0.01, 0.05, 0.1]},
       pre_dispatch='2*n_jobs', refit=True, scoring='f1', verbose=0)
	Accuracy: 0.71847	Precision: 0.26887	Recall: 0.64650	F1: 0.37979	F2: 0.50472
	Total predictions: 15000	True positives: 1293	False positives: 3516	False negatives:  707	True negatives: 9484
```

No suprise, there are no good classifier using all features.

# Remove the features above 70% nan percent

    deferral_payments nan percentile: 0.7379310344827587
    loan_advances nan percentile: 0.9793103448275862
    restricted_stock_deferred nan percentile: 0.8827586206896552
    director_fees nan percentile: 0.8896551724137931

# Visualize features distribution

![png](./plot/output_27_1.png)


![png](./plot/output_27_3.png)


![png](./plot/output_27_11.png)


From the deferred_income bar chart I find that the deferred_income should be transform to positive values so as to keep align with other features.

```python
df["deferred_income"] = df["deferred_income"].abs()
```

After looking through the bar chart, I decide to remove 'from_messages', 'to_messages', 'total_payments' since they are noisy and having outliers.

# Plot Correlation Matrix

![finance_features_correlation_matrix](./plot/finance_features_correlation_matrix.png)


![email_features_correlation_matrix](./plot/email_features_correlation_matrix.png)



# PCA
From the correlation matrix we can see that "exercised_stock_options", "restricted_stock", "total_stock_value" are highly related. So I decide to use PCA to transform them into one new feature.

# Test again

```
    Fitting the classifier to the training set
    done in 13.233s
    RandomForestClassifier(bootstrap=True, class_weight='balanced',
                criterion='gini', max_depth=None, max_features='auto',
                max_leaf_nodes=None, min_impurity_decrease=0.0,
                min_impurity_split=None, min_samples_leaf=2,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                n_estimators=50, n_jobs=-1, oob_score=True, random_state=42,
                verbose=0, warm_start=False)
    Predicting on the training set
    Done in 0.107s
                 precision    recall  f1-score   support

          False       1.00      0.99      0.99        87
           True       0.93      1.00      0.97        14

    avg / total       0.99      0.99      0.99       101

    [[86  1]
     [ 0 14]]
    Predicting on the testing set
    Done in 0.107s
                 precision    recall  f1-score   support

          False       0.95      0.97      0.96        40
           True       0.67      0.50      0.57         4

    avg / total       0.93      0.93      0.93        44

    [[39  1]
     [ 2  2]]
    RandomForestClassifier(bootstrap=True, class_weight='balanced',
                criterion='gini', max_depth=None, max_features='auto',
                max_leaf_nodes=None, min_impurity_decrease=0.0,
                min_impurity_split=None, min_samples_leaf=2,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                n_estimators=50, n_jobs=-1, oob_score=True, random_state=42,
                verbose=0, warm_start=False)
    	Accuracy: 0.86707	Precision: 0.50294	Recall: 0.25650	F1: 0.33974	F2: 0.28437
    	Total predictions: 15000	True positives:  513	False positives:  507	False negatives: 1487	True negatives: 12493

    Fitting the classifier to the training set
    done in 25.353s
    AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
              learning_rate=0.1, n_estimators=125, random_state=42)
    Predicting on the training set
    Done in 0.020s
                 precision    recall  f1-score   support

          False       0.98      1.00      0.99        87
           True       1.00      0.86      0.92        14

    avg / total       0.98      0.98      0.98       101

    [[87  0]
     [ 2 12]]
    Predicting on the testing set
    Done in 0.018s
                 precision    recall  f1-score   support

          False       0.95      0.93      0.94        40
           True       0.40      0.50      0.44         4

    avg / total       0.90      0.89      0.89        44

    [[37  3]
     [ 2  2]]
    AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
              learning_rate=0.1, n_estimators=125, random_state=42)
    	Accuracy: 0.87207	Precision: 0.53344	Recall: 0.32300	F1: 0.40237	F2: 0.35067
    	Total predictions: 15000	True positives:  646	False positives:  565	False negatives: 1354	True negatives: 12435

    Fitting the classifier to the training set
    done in 2.856s
    LogisticRegression(C=5, class_weight='balanced', dual=False,
              fit_intercept=True, intercept_scaling=1, max_iter=100,
              multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
              solver='liblinear', tol=1e-05, verbose=0, warm_start=False)
    Predicting on the training set
    Done in 0.000s
                 precision    recall  f1-score   support

          False       0.95      0.80      0.87        87
           True       0.37      0.71      0.49        14

    avg / total       0.87      0.79      0.82       101

    [[70 17]
     [ 4 10]]
    Predicting on the testing set
    Done in 0.000s
                 precision    recall  f1-score   support

          False       0.97      0.88      0.92        40
           True       0.38      0.75      0.50         4

    avg / total       0.92      0.86      0.88        44

    [[35  5]
     [ 1  3]]
    GridSearchCV(cv=None, error_score='raise',
           estimator=LogisticRegression(C=1.0, class_weight='balanced', dual=False,
              fit_intercept=True, intercept_scaling=1, max_iter=100,
              multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
              solver='liblinear', tol=0.0001, verbose=0, warm_start=False),
           fit_params={}, iid=True, n_jobs=1,
           param_grid={'C': [0.1, 0.5, 1, 5, 10, 50, 100], 'tol': [1e-05, 5e-05, 0.0001, 0.0005, 0.001, 0.005, 0.01], 'max_iter': [100, 150, 200, 250, 300]},
           pre_dispatch='2*n_jobs', refit=True, scoring='f1', verbose=0)
    	Accuracy: 0.74360	Precision: 0.28313	Recall: 0.60250	F1: 0.38523	F2: 0.49160
    	Total predictions: 15000	True positives: 1205	False positives: 3051	False negatives:  795	True negatives: 9949

    Fitting the classifier to the training set
    done in 0.515s
    Best estimator found by grid search:
    SVC(C=0.5, cache_size=200, class_weight='balanced', coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)
    Predicting on the training set
    Done in 0.001s
                 precision    recall  f1-score   support

          False       0.93      0.76      0.84        87
           True       0.30      0.64      0.41        14

    avg / total       0.84      0.74      0.78       101

    [[66 21]
     [ 5  9]]
    Predicting on the testing set
    Done in 0.000s
                 precision    recall  f1-score   support

          False       1.00      0.75      0.86        40
           True       0.29      1.00      0.44         4

    avg / total       0.94      0.77      0.82        44

    [[30 10]
     [ 0  4]]
    GridSearchCV(cv=None, error_score='raise',
           estimator=SVC(C=1.0, cache_size=200, class_weight='balanced', coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False),
           fit_params={}, iid=True, n_jobs=1,
           param_grid={'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50], 'gamma': [0.001, 0.005, 0.01, 0.05, 0.1]},
           pre_dispatch='2*n_jobs', refit=True, scoring='f1', verbose=0)
    	Accuracy: 0.72040	Precision: 0.27419	Recall: 0.66600	F1: 0.38845	F2: 0.51797
    	Total predictions: 15000	True positives: 1332	False positives: 3526	False negatives:  668	True negatives: 9474
```



# Sort features importance

Use adaboost classifier to sort features by their importances.

    [ 0.216  0.216  0.136  0.112  0.112  0.088  0.08   0.016  0.016  0.008]
    ['bonus' 'other' 'expenses' 'salary' 'fraction_from_this_person_to_poi'
     'fraction_from_poi_to_this_person' 'stock_pca' 'deferred_income'
     'fraction_shared_receipt_with_poi' 'long_term_incentive']

    array(['bonus', 'other', 'expenses', 'salary',
           'fraction_from_this_person_to_poi',
           'fraction_from_poi_to_this_person', 'stock_pca', 'deferred_income',
           'fraction_shared_receipt_with_poi', 'long_term_incentive'],
          dtype='<U32')

Since the last three features 'deferred_income', 'fraction_shared_receipt_with_poi', 'long_term_incentive are relatively unimportant, I try to remove them.


# Final test

```
    Fitting the classifier to the training set
    done in 14.340s
    RandomForestClassifier(bootstrap=True, class_weight='balanced',
                criterion='gini', max_depth=None, max_features='auto',
                max_leaf_nodes=None, min_impurity_decrease=0.0,
                min_impurity_split=None, min_samples_leaf=2,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                n_estimators=150, n_jobs=-1, oob_score=True, random_state=42,
                verbose=0, warm_start=False)
    Predicting on the training set
    Done in 0.106s
                 precision    recall  f1-score   support

          False       1.00      1.00      1.00        87
           True       1.00      1.00      1.00        14

    avg / total       1.00      1.00      1.00       101

    [[87  0]
     [ 0 14]]
    Predicting on the testing set
    Done in 0.106s
                 precision    recall  f1-score   support

          False       0.95      0.97      0.96        40
           True       0.67      0.50      0.57         4

    avg / total       0.93      0.93      0.93        44

    [[39  1]
     [ 2  2]]
    RandomForestClassifier(bootstrap=True, class_weight='balanced',
                criterion='gini', max_depth=None, max_features='auto',
                max_leaf_nodes=None, min_impurity_decrease=0.0,
                min_impurity_split=None, min_samples_leaf=2,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                n_estimators=150, n_jobs=-1, oob_score=True, random_state=42,
                verbose=0, warm_start=False)
    	Accuracy: 0.86847	Precision: 0.51192	Recall: 0.29000	F1: 0.37025	F2: 0.31753
    	Total predictions: 15000	True positives:  580	False positives:  553	False negatives: 1420	True negatives: 12447

    Fitting the classifier to the training set
    done in 25.123s
    AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
              learning_rate=0.1, n_estimators=125, random_state=42)
    Predicting on the training set
    Done in 0.020s
                 precision    recall  f1-score   support

          False       0.99      1.00      0.99        87
           True       1.00      0.93      0.96        14

    avg / total       0.99      0.99      0.99       101

    [[87  0]
     [ 1 13]]
    Predicting on the testing set
    Done in 0.019s
                 precision    recall  f1-score   support

          False       0.95      0.93      0.94        40
           True       0.40      0.50      0.44         4

    avg / total       0.90      0.89      0.89        44

    [[37  3]
     [ 2  2]]
    AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
              learning_rate=0.1, n_estimators=125, random_state=42)
    	Accuracy: 0.86287	Precision: 0.48008	Recall: 0.34350	F1: 0.40047	F2: 0.36422
    	Total predictions: 15000	True positives:  687	False positives:  744	False negatives: 1313	True negatives: 12256

    Fitting the classifier to the training set
    done in 2.747s
    LogisticRegression(C=0.5, class_weight='balanced', dual=False,
              fit_intercept=True, intercept_scaling=1, max_iter=100,
              multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
              solver='liblinear', tol=1e-05, verbose=0, warm_start=False)
    Predicting on the training set
    Done in 0.001s
                 precision    recall  f1-score   support

          False       0.95      0.80      0.87        87
           True       0.37      0.71      0.49        14

    avg / total       0.87      0.79      0.82       101

    [[70 17]
     [ 4 10]]
    Predicting on the testing set
    Done in 0.000s
                 precision    recall  f1-score   support

          False       1.00      0.88      0.93        40
           True       0.44      1.00      0.62         4

    avg / total       0.95      0.89      0.90        44

    [[35  5]
     [ 0  4]]
    GridSearchCV(cv=None, error_score='raise',
           estimator=LogisticRegression(C=1.0, class_weight='balanced', dual=False,
              fit_intercept=True, intercept_scaling=1, max_iter=100,
              multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
              solver='liblinear', tol=0.0001, verbose=0, warm_start=False),
           fit_params={}, iid=True, n_jobs=1,
           param_grid={'C': [0.1, 0.5, 1, 5, 10, 50, 100], 'tol': [1e-05, 5e-05, 0.0001, 0.0005, 0.001, 0.005, 0.01], 'max_iter': [100, 150, 200, 250, 300]},
           pre_dispatch='2*n_jobs', refit=True, scoring='f1', verbose=0)
    	Accuracy: 0.75300	Precision: 0.31639	Recall: 0.73450	F1: 0.44227	F2: 0.58095
    	Total predictions: 15000	True positives: 1469	False positives: 3174	False negatives:  531	True negatives: 9826

    Fitting the classifier to the training set
    done in 0.511s
    Best estimator found by grid search:
    SVC(C=5, cache_size=200, class_weight='balanced', coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma=0.05, kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)
    Predicting on the training set
    Done in 0.001s
                 precision    recall  f1-score   support

          False       0.94      0.76      0.84        87
           True       0.32      0.71      0.44        14

    avg / total       0.86      0.75      0.79       101

    [[66 21]
     [ 4 10]]
    Predicting on the testing set
    Done in 0.000s
                 precision    recall  f1-score   support

          False       1.00      0.75      0.86        40
           True       0.29      1.00      0.44         4

    avg / total       0.94      0.77      0.82        44

    [[30 10]
     [ 0  4]]
    GridSearchCV(cv=None, error_score='raise',
           estimator=SVC(C=1.0, cache_size=200, class_weight='balanced', coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False),
           fit_params={}, iid=True, n_jobs=1,
           param_grid={'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50], 'gamma': [0.001, 0.005, 0.01, 0.05, 0.1]},
           pre_dispatch='2*n_jobs', refit=True, scoring='f1', verbose=0)
    	Accuracy: 0.72587	Precision: 0.28718	Recall: 0.71250	F1: 0.40937	F2: 0.54968
    	Total predictions: 15000	True positives: 1425	False positives: 3537	False negatives:  575	True negatives: 9463
```


This random forest classifier works good in classification_report and confusion_matrix from sklearn.metrics.
But weak in the test_classifier using StratifiedShuffleSplit.

After remove three features, the adaboost classifier down 0.5 on precision and up 0.2 on recall.

This logistec classifier have amazing recall score.
It is useful when we try to catch more guilty.


# Final Decision
After comparing, I decide use adaboost classifier on
['bonus' 'other' 'expenses' 'salary' 'fraction_from_this_person_to_poi'
 'fraction_from_poi_to_this_person' 'stock_pca' 'deferred_income'
 'fraction_shared_receipt_with_poi' 'long_term_incentive']

    AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
              learning_rate=0.1, n_estimators=125, random_state=42)
    Predicting on the training set
    Done in 0.020s
                 precision    recall  f1-score   support

          False       0.98      1.00      0.99        87
           True       1.00      0.86      0.92        14

    avg / total       0.98      0.98      0.98       101

    [[87  0]
     [ 2 12]]
    Predicting on the testing set
    Done in 0.018s
                 precision    recall  f1-score   support

          False       0.95      0.93      0.94        40
           True       0.40      0.50      0.44         4

    avg / total       0.90      0.89      0.89        44

    [[37  3]
     [ 2  2]]
    AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
              learning_rate=0.1, n_estimators=125, random_state=42)
    	Accuracy: 0.87207	Precision: 0.53344	Recall: 0.32300	F1: 0.40237	F2: 0.35067
    	Total predictions: 15000	True positives:  646	False positives:  565	False negatives: 1354	True negatives: 12435


# Dump classifier, dataset, and features_list
```python
with open('my_classifier.pkl', 'wb') as c:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(clf, c, pickle.HIGHEST_PROTOCOL)

with open('my_dataset.pkl', 'wb') as m:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(my_dataset, m, pickle.HIGHEST_PROTOCOL)

with open('my_feature_list.pkl', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(features_list, f, pickle.HIGHEST_PROTOCOL)
```

# Reflexion

Although I write only three classifier test in this markdown document. But actually,
I done it hundreds of times.

Since this is an imbalanced labeled and small dataset, it is hard to test whether a
classifier is good.

In training process, many classifier get amazing score in classification_report
and confusion_matrix, but just gain disappointed score using StratifiedShuffleSplit.

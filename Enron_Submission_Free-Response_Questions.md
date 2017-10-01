#### Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]

  This project is aimed to identified the person of interest in Enron scandal.

  In 2000, Enron was one of the largest companies in the United States.
 By 2002, it had collapsed into bankruptcy due to widespread corporate fraud.
 In the resulting Federal investigation, a significant amount of typically
 confidential information entered into the public record, including tens of
 thousands of emails and detailed financial data for top executives.  

  In this project, I will try to build a detecitve model to identify the
 person of interest in Enron scandal, based on financial and email data
 made public as a result of the Enron scandal.

  The finance and email data will give some important hints about who is
the people of interest. But we can not simply find them according to some simple
conditions such as does he have salary above what level or sent more than how
many email.  

  Since there are dozens of features would be used to identify the person of
interest and this is an imbalance classification problem, machine learning would
be helpful to deal with such a complicated problem.  

  In the exploring process, I found there is one outlier came from the finance
data, the total. Of course total is nobody related with people of interest so
that it should be removed. What is more, there are also some outliers in the
dataset, but it turns out that they are the big man in Enron.Therefore, they
should be kept.  
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

Here is the dataset overview:

  characteristics | overview
  --- | ---
  Total number of persons| 146
  Total number of persons of interest (POI) | 18
  Total number of non persons of interest (non POI) | 128
  Total number of features | 19
  Total number of finance features | 14
  Total number of email features | 5

Features missing percent:

  features | percent
  --- | ---
  salary |  0.35172413793103446
  deferral_payments |  0.7379310344827587
  total_payments |  0.14482758620689656
  loan_advances |  0.9793103448275862
  bonus |  0.4413793103448276
  restricted_stock_deferred |  0.8827586206896552
  deferred_income |  0.6689655172413793
  total_stock_value |  0.13793103448275862
  expenses |  0.35172413793103446
  exercised_stock_options |  0.30344827586206896
  other |  0.36551724137931035
  long_term_incentive |  0.5517241379310345
  restricted_stock |  0.2482758620689655
  director_fees |  0.8896551724137931
  to_messages |  0.4068965517241379
  from_messages |  0.4068965517241379
  from_poi_to_this_person |  0.0
  from_this_person_to_poi |  0.0
  shared_receipt_with_poi |  0.0

#### What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “intelligently select features”, “properly scale features”]

The features I used finally are
  * bonus
  * other
  * expenses
  * salary
  * fraction_from_this_person_to_poi
  * fraction_from_poi_to_this_person
  * stock_pca
  * fraction_shared_receipt_with_poi

At first, I make three new features by transforming

  * from_poi_to_this_person
  * from_this_person_to_poi
  * shared_receipt_with_poi

to percent of the according total email number and
removed these three original features. Because the percent can better indicate
the relationship between the people with the people of interest.

  After that, I just check the nan percent of every features and removed
remove the features above 70% nan percent. Obviously, too many nan value would
make a feature less useful unless it have strong exclusive relationship with
one of the label. Actually, I chose 50% threshold at first and I found that the
70% threshold performs as good as the 50%, both get 0.91 f1 score in classification_report
on test set using random forest. But I decided to keep more features for further testing.
What is more, the distribution of two features with nan percent between 0.5 and 0.7 seems useful.
Actually, in the end, all the features having nan percent higher than 0.5 were removed.
But I just want to do it step by step to slowly explore the features.

  And then I visualize the feature distribution to cross-validate that the
features with high nan percent are useless and remove 'from_messages',
'to_messages', 'total_payments' since they are all sum, noisy and having outliers.

  Further more, I apply minmax scaling on all finance features because of
their various range.

  What is more, I apply PCA on "exercised_stock_options",
"restricted_stock", "total_stock_value" into a new feature 'stock_pca'
since they are highly correlated. If the PCA applied on all features, the
result just not good, no matter how many Principal components are used because
the first principal component already large enough.

  When I use the random forest classifier on the features, I get the feature
importances as below:  

  features | importances
  ---- | ----
  other | 0.20126948
  expenses | 0.18295673
  fraction_from_this_person_to_poi | 0.14281353
  stock_pca | 0.1413511
  bonus | 0.09962223
  salary | 0.06444563
  fraction_from_poi_to_this_person | 0.05627843
  deferred_income | 0.04823193
  fraction_shared_receipt_with_poi | 0.03722525
  long_term_incentive | 0.02580569

  While using the adaboost classifier on the features, I get the feature
importances as below:  

  features | importances
  ---- | ----
  bonus | 0.32
  other | 0.28
  stock_pca | 0.12
  expenses | 0.08
  fraction_from_this_person_to_poi | 0.08
  salary | 0.06
  fraction_shared_receipt_with_poi | 0.06
  deferred_income | 0.0
  long_term_incentive | 0.0
  fraction_from_poi_to_this_person | 0.0

  I noticed that both long_term_incentive and deferred_income are appeared on the last three importances in both random forest and adaboost. Consequently and last but not least, I remove them and make my final features.

  Of course, I also try automated feature selection process like k-best, the result is
  not bad- f1 score near 0.4, but just cannot compare with the final result I get. Since I have enough time
  to explore the relation between feature and trained model, I prefer to get my had dirty.

#### What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]

  I used random forest classifier in the end. And I also try adaboost, logistic,
naive bayes, svm and knn.

adaboost works as good as random forest in classification_report and
confusion_matrix from sklearn.metrics, but weak in the test_classifier in final
test using StratifiedShuffleSplit. The precision is as good as random forest,
but the recall is just a little above 0.3.

  This logistec classifier have precision just near about and amazing recall that
about 0.7. It is useful when we try to catch more guilty.  

  As for the SVM and naive bayes, both of them get precision and recall just about 0.3.

  The knn is the worst classifier for this problem, both precision and recall cannot even near 0.1.

#### What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? What parameters did you tune? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric items: “discuss parameter tuning”, “tune the algorithm”]

  Tuning parameters can help preventing overfit and get better performance on
testing set. I just use the GridSearchCV to tune my classifier.  

  As for the final classifier I chose, the random forest classifier, I tuned the
n_estimators and max_depth. At first, I just tune the n_estimators and min_samples_leaf,
and the recall is very close but below the 0.3. When I tuned the max_depth down to 5,
the recall start higher than 0.3, and after I tuning it down to 3, the recall surged
to above 0.5 and the precision still keep above 0.5.    

  As for the others, I tune n_estimators and learning_rate in adaboost, C in logistic, C, gamma in svm, n_neighbors in knn.   

#### What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric items: “discuss validation”, “validation strategy”]

  Validation is applying the trained model on testing data set, and evaluating
its performance.

A classic mistake is overfitting. At first, all my classifier
get poor performance on testing set. After long time parameters tuning, they
still have no any improvement. Finally, after I print the validation score on
training set, I realized that they are overfitted and the reason is that I used
default accuracy score to train the model. Since the dataset is imbalanced and
just a few sample, I changed the scoring method to F1 score. Accordingly, the
classifier performance improved instantly.

  I first validate my analysis by applying classification_report on testing set.
If the result is good, I will do further validation by applying
StratifiedShuffleSplit on whole dataset because of the long time it took and the
nature of the dataset, small and imbalanced.

#### Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]

  Actually, I used F1 score to train the model and in fact, the f1 score are
calculated by the precision and recall. According to the presumption of
innocence, I think the random forest have best performance, with 0.53081
Precision and 0.56000 Recall. It is the only one classfier that both precision and recall higher than 0.5.  

  In human words, in 100 POIs that predictd by the model, there would be 53 true
POIs. On the other hand, in 100 true POIs, the model can determine 56 of them.

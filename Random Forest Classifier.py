import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords #stopwords library
import pickle
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt


# -------------------- LOADING DATA -------------
manifesto_project = pd.read_csv("G:\GitHub\FINAL PROJECT\Political-Bias-NLP\manifesto_clean.csv")

manifesto_project.dtypes

manifesto_project['processed_text'] = manifesto_project['processed_text'].astype(str)


manifesto_project.rename(columns={'Unnamed: 0':'Index'}, inplace=True)
manifesto_project= manifesto_project.set_index('Index')
manifesto_project['cmp_code'] = manifesto_project['cmp_code'].astype('int')
manifesto_project['domain_name'] = manifesto_project['domain_name'].astype('category')


# ----------------------------------------


X = manifesto_project.iloc[:, 5:9]

Y = manifesto_project.iloc[:,10].values


x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=42)


# tfidf_vectorizer = TfidfVectorizer()

# x_train_tfidf = tfidf_vectorizer.fit_transform(x_train) # using fit_transform on train data
# # x_train_tfidf.shape

# x_test_tfidf = tfidf_vectorizer.transform(x_test)

# tfidfconverter = TfidfVectorizer(max_features=2000, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))  
# x = tfidfconverter.fit_transform(X).toarray()

# ------------------------ Simple Random Forest Classifier -----------

model = RandomForestClassifier(random_state=1, n_estimators=1200, max_depth=30, 
                               min_samples_split=100, min_samples_leaf=2)


model.fit(x_train, y_train)

y_pred = model.predict(x_test)


print(confusion_matrix(y_pred, y_test))

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# [[12238  7055  6040]
#  [  311  1793   136]
#  [   19    32   304]]
# 0.5132841592666858
#               precision    recall  f1-score   support

#          0.0       0.48      0.97      0.65     12568
#          1.0       0.80      0.20      0.32      8880
#          2.0       0.86      0.05      0.09      6480

#     accuracy                           0.51     27928
#    macro avg       0.71      0.41      0.35     27928
# weighted avg       0.67      0.51      0.41     27928

# ---------------------- testing parameters for RandomForestClassifier with sentiment analysis, results commented below
n_estimators = [500, 800, 1200]
max_depth = [15, 25, 30]
min_samples_split = [10, 15, 100]
min_samples_leaf = [2, 5, 10] 

forest = RandomForestClassifier(random_state = 1)

hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,  
              min_samples_split = min_samples_split, 
             min_samples_leaf = min_samples_leaf)

gridF = GridSearchCV(forest, hyperF, cv = 3, verbose = 3, 
                      n_jobs = -1)
bestF = gridF.fit(x_train, y_train)


results = bestF.cv_results_

print(results.params)

print (bestF.best_params_)



results.score


# {'max_depth': 30, 'min_samples_leaf': 2, 'min_samples_split': 10, 'n_estimators': 500}

# ---------------------------------------------- best parameters for RandomForest with vectorized text


model = RandomForestClassifier(random_state=1, n_estimators=500, max_depth=30, 
                               min_samples_split=10, min_samples_leaf=2)


model.fit(x_train_tfidf, y_train)

y_pred = model.predict(x_test_tfidf)


print(confusion_matrix(y_pred, y_test))

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# [[12217  6993  6026]
#  [  328  1851   145]
#  [   23    36   309]]
# 0.5147880263534804
#               precision    recall  f1-score   support

#          0.0       0.48      0.97      0.65     12568
#          1.0       0.80      0.21      0.33      8880
#          2.0       0.84      0.05      0.09      6480

#     accuracy                           0.51     27928
#    macro avg       0.71      0.41      0.36     27928
# weighted avg       0.67      0.51      0.42     27928


#  -------------------

model = RandomForestClassifier(max_depth= 15, min_samples_leaf= 2, min_samples_split= 100, n_estimators= 500)


model.fit(x_train, y_train)

y_pred = model.predict(x_test)


print(confusion_matrix(y_pred, y_test))

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
# [[11362  7668  5227]
#  [  656   725   454]
#  [  550   487   799]]
# 0.46140074477227155
#               precision    recall  f1-score   support

#          0.0       0.47      0.90      0.62     12568
#          1.0       0.40      0.08      0.14      8880
#          2.0       0.44      0.12      0.19      6480

#     accuracy                           0.46     27928
#    macro avg       0.43      0.37      0.31     27928
# weighted avg       0.44      0.46      0.37     27928
















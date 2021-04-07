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
from sklearn.ensemble import GradientBoostingRegressor
from numpy import mean
from numpy import std
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score




# -------------------- LOADING DATA -------------
manifesto_project = pd.read_csv("G:\GitHub\FINAL PROJECT\Political-Bias-NLP\Manifesto_Project_processed.csv")

manifesto_project.dtypes

manifesto_project['processed_text'] = manifesto_project['processed_text'].astype(str)


manifesto_project.rename(columns={'Unnamed: 0':'Index'}, inplace=True)
manifesto_project= manifesto_project.set_index('Index')
# manifesto_project['cmp_code'] = manifesto_project['cmp_code'].astype('category')
# manifesto_project['label'] = manifesto_project['label'].astype('category')
manifesto_project['domain_name'] = manifesto_project['domain_name'].astype('category')
manifesto_project['variable_name'] = manifesto_project['variable_name'].astype('category')
manifesto_project['domain_code'] = manifesto_project['domain_code'].astype('category')



manifesto_project.isna().sum()

manifesto_project.label.unique()



# ----------------- Train and Test ------------------
X = manifesto_project.iloc[:, 1].values 

Y = manifesto_project.iloc[:,2].values


x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

# --------- Vectorization of X

tfidf_vectorizer = TfidfVectorizer()

x_train_tfidf = tfidf_vectorizer.fit_transform(x_train) # using fit_transform on train data
# x_train_tfidf.shape

x_test_tfidf = tfidf_vectorizer.transform(x_test)

tfidfconverter = TfidfVectorizer(max_features=2000, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))  
x = tfidfconverter.fit_transform(X).toarray()


# ------------- Multinomial Logistic Regression ---------------
# RUNNING THIS LAGS MY PC. DO NOT RUN

# evaluate multinomial logistic regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')

model.fit(x_train_tfidf, y_train)

# # define the model evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# evaluate the model and collect the scores
n_scores = cross_val_score(model, x, Y, scoring='accuracy', cv=cv, n_jobs=-1)
# report the model performance
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

# ------------------ TFDF + RandomForest ---------------------- 
 
from sklearn.ensemble import RandomForestClassifier
text_classifier = RandomForestClassifier(n_estimators=400, random_state=0)  
text_classifier.fit(x_train_tfidf, y_train)
 
 
y_pred = text_classifier.predict(x_test)

# -------------------- Gradient Boosting ----------------

X = manifesto_project.iloc[:, 1].values 

Y = manifesto_project.iloc[:,2].values

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer()

x_train_tfidf = tfidf_vectorizer.fit_transform(x_train) 

x_test_tfidf = tfidf_vectorizer.transform(x_test)

tfidfconverter = TfidfVectorizer(max_features=2000, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))  
x = tfidfconverter.fit_transform(X).toarray()



gbrt = GradientBoostingRegressor(random_state = 42, 
                                 learning_rate = 0.15, 
                                 max_depth = 9, 
                                 n_estimators = 1000, 
                                 min_samples_leaf = 50,
                                 min_samples_split = 1300)
gbrt.fit(x_train_tfidf, y_train)

print("Accuracy on training set: ", gbrt.score(x_train_tfidf, y_train))
print("Accuracy on test set: ", gbrt.score(x_test_tfidf, y_test))

y_pred = gbrt.predict(x_test_tfidf)


# with learning_rate = 0.5, 450 estimators, min_samples_leaf=5, max_depth = 6:
# Accuracy on training set:  0.5712684783459708
# Accuracy on test set:  0.29151574988961715

# with learning_rate = 0.1, 450 estimators, min_samples_leaf=5, max_depth = 6:
# Accuracy on training set:  0.4106127997986475
# Accuracy on test set:  0.2769594619404574

# with learning_rate = 0.1, 700 estimators, min_samples_leaf=5, max_depth = 6:
# Accuracy on training set:  0.45616150699763713
# Accuracy on test set:  0.2945183429372382

# with learning_rate = 0.5, 700 estimators, min_samples_leaf=5, max_depth = 6:
# Accuracy on training set:  0.6222019301742178
# Accuracy on test set:  0.2901307862808413

# with learning_rate = 0.7, 700 estimators, min_samples_leaf=5, max_depth = 6:
# Accuracy on training set:  0.6561875988846932
# Accuracy on test set:  0.24591437366545643


# with learning_rate = 0.7, 700 estimators, min_samples_leaf=2, max_depth = 6:
# Accuracy on training set:  0.7021165421146279
# Accuracy on test set:  0.2086118861691315

# with learning_rate = 0.1, 1000 estimators, min_samples_leaf=3, max_depth = 6:
# Accuracy on training set:  0.5176065074459419
# Accuracy on test set:  0.2982450811408234

# with learning_rate = 0.15, 1000 estimators, min_samples_leaf=50, max_depth = 6, min_samples_split = 1300:
# Accuracy on training set:  0.3753214828075757
# Accuracy on test set:  0.28850052566433537


# with learning_rate = 0.15, 1000 estimators, min_samples_leaf=50, max_depth = 9, min_samples_split = 1300:


# --------------- K-Nearest Neighbours --------------




# ----------------- Decision Trees ---------------



#  ---------------- Naive Bayes ----------------------










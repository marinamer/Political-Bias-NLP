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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import scipy


# -------------------- LOADING DATA -------------
manifesto_project = pd.read_csv("G:\GitHub\FINAL_PROJECT\Political-Bias-NLP\manifesto_clean.csv")

manifesto_project.dtypes

manifesto_project['processed_text'] = manifesto_project['processed_text'].astype(str)


manifesto_project.rename(columns={'Unnamed: 0':'Index'}, inplace=True)
manifesto_project= manifesto_project.set_index('Index')
manifesto_project['cmp_code'] = manifesto_project['cmp_code'].astype('int')
# manifesto_project['label'] = manifesto_project['label'].astype('category')
manifesto_project['domain_name'] = manifesto_project['domain_name'].astype('category')


manifesto_project.dtypes

# # ----------------------------------------


X = manifesto_project.iloc[:40000]['processed_text']


Y = manifesto_project.iloc[:40000]['bias_numeric']
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer()

x_train_tfidf = tfidf_vectorizer.fit_transform(x_train) # using fit_transform on train data
# x_train_tfidf.shape

# import pickle
# filename = 'tfidftest.pkl'
# pickle.dump(tfidf_vectorizer, open(filename, 'wb'))         

x_test_tfidf = tfidf_vectorizer.transform(x_test)



# explicitly require this experimental feature
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
# now you can import normally from ensemble
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords


# Import Support Vector Classifier
from sklearn.svm import SVC
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
svc=SVC(probability=True, kernel='linear')

# Create adaboost classifer object
abc =AdaBoostClassifier(n_estimators=3, base_estimator=svc,learning_rate=1)

# Train Adaboost Classifer
model = abc.fit(x_train_tfidf, y_train)

#Predict the response for test dataset
y_pred = model.predict(x_test_tfidf)


# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#Accuracy(3 estimators): 0.764
# Accuracy(5 estimators): 0.684375
#Accuracy(4 estimators): 0.7255


# -----------------------

pickle.dump(abc, open('Ada3est76acc.sav', 'wb'))
















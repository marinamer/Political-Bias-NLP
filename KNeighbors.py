import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
import pandas as pd
from nltk.corpus import stopwords #stopwords library
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

import scipy

# -------------------- LOADING DATA -------------
# manifesto_project = pd.read_csv("G:\GitHub\FINAL PROJECT\Political-Bias-NLP\Manifesto_Project_processed.csv")

manifesto_project = pd.read_csv("G:\GitHub\FINAL_PROJECT\Political-Bias-NLP\manifesto_clean.csv")

manifesto_project.dtypes

manifesto_project['processed_text'] = manifesto_project['processed_text'].astype(str)


manifesto_project.rename(columns={'Unnamed: 0':'Index'}, inplace=True)
manifesto_project= manifesto_project.set_index('Index')
manifesto_project['cmp_code'] = manifesto_project['cmp_code'].astype('int')
# manifesto_project['label'] = manifesto_project['label'].astype('category')
manifesto_project['domain_name'] = manifesto_project['domain_name'].astype('category')
# manifesto_project['variable_name'] = manifesto_project['variable_name'].astype('category')
# manifesto_project['domain_code'] = manifesto_project['domain_code'].astype('category')


manifesto_project.dtypes



# left =  [103, 105, 106, 107, 202, 403, 404, 406, 412, 413, 504, 506, 701]
# right = [104, 201, 203, 305, 401, 402, 407, 414, 505, 601, 603, 605, 606]


# def bias(c):
#     if c['cmp_code'] in left:
#         return "left_wing"
#     elif c['cmp_code'] in right:
#         return "right_wing"
#     else:
#         return "neutral"



# manifesto_project['bias'] = manifesto_project.apply(bias, axis=1)

# manifesto_project.loc[manifesto_project['bias'] == "left_wing", 'bias_numeric'] = 1
# manifesto_project.loc[manifesto_project['bias'] == "right_wing", 'bias_numeric'] = 2
# manifesto_project.loc[manifesto_project['bias'] == "neutral", 'bias_numeric'] = 0

# def sent_to_words(sentences):
#     for sentence in sentences:
#         yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

# manifesto_project['processed_text_token'] = list(sent_to_words(manifesto_project['processed_text']))

# print(manifesto_project['processed_text_token'][:1])


# tfidf_vectorizer = TfidfVectorizer()
# tfidfconverter = TfidfVectorizer(max_features=2000, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
# manifesto_project['tfidf'] = tfidfconverter.fit_transform(manifesto_project['processed_text'])

# manifesto_project['tfidf_array'] = scipy.sparse.csr_matrix.toarray(manifesto_project['tfidf'])

# ----------------------------------------


X = manifesto_project.iloc[:, 1].values 

Y = manifesto_project.iloc[:,10].values


x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=42)


tfidf_vectorizer = TfidfVectorizer()

x_train_tfidf = tfidf_vectorizer.fit_transform(x_train) # using fit_transform on train data
# x_train_tfidf.shape

x_test_tfidf = tfidf_vectorizer.transform(x_test)

tfidfconverter = TfidfVectorizer(max_features=2000, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))  
x = tfidfconverter.fit_transform(X).toarray()


# we need to look into parameters to see what works best. 

# model = KNeighborsClassifier(n_neighbors=3)

# model.fit(x_train_tfidf, y_train)

# y_pred = model.predict(x_test_tfidf)


# print(confusion_matrix(y_pred, y_test))

# print(accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))


# With these settings we got a 0.636 accuracy (sent to Luke)

# print(confusion_matrix(y_pred, y_test))
# [[9474 2881 2370]
#  [1768 5088  904]
#  [1326  911 3206]]

# print(accuracy_score(y_test, y_pred))
# 0.6362073904325408

# print(classification_report(y_test, y_pred))
#               precision    recall  f1-score   support

#            0       0.64      0.75      0.69     12568
#            1       0.66      0.57      0.61      8880
#            2       0.59      0.49      0.54      6480

#     accuracy                           0.64     27928
#    macro avg       0.63      0.61      0.61     27928
# weighted avg       0.63      0.64      0.63     27928




# manifesto_project.to_csv("G:\GitHub\FINAL PROJECT\Political-Bias-NLP\manifesto_clean.csv")


# ----------------------------


#List Hyperparameters that we want to tune.
leaf_size = [1, 10, 15, 25, 50]
n_neighbors = [3, 5, 15, 20, 30]
p=[1,2]
#Convert to dictionary
hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
#Create new KNN object
knn_2 = KNeighborsClassifier()
#Use GridSearch
clf = GridSearchCV(knn_2, hyperparameters, cv=10, verbose=3)
#Fit the model
best_model = clf.fit(x_train_tfidf, y_train)
#Print The value of best Hyperparameters
print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
print('Best p:', best_model.best_estimator_.get_params()['p'])
print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])

# Best leaf_size: 1
# Best p: 2
# Best n_neighbors: 30



print(confusion_matrix(best_model, y_test))

print(accuracy_score(y_test, best_model))
print(classification_report(y_test, best_model))



print(best_model.cv_results_)

# -------- Best Model -----------

model = KNeighborsClassifier(leaf_size= 1,n_neighbors=30, p=2)

model.fit(x_train_tfidf, y_train)

y_pred = model.predict(x_test_tfidf)


print(confusion_matrix(y_pred, y_test))

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# [[9894 2655 2415]
#  [1717 5527 1037]
#  [ 957  698 3028]]
# 0.6605915210541392
#               precision    recall  f1-score   support

#          0.0       0.66      0.79      0.72     12568
#          1.0       0.67      0.62      0.64      8880
#          2.0       0.65      0.47      0.54      6480

#     accuracy                           0.66     27928
#    macro avg       0.66      0.63      0.64     27928
# weighted avg       0.66      0.66      0.65     27928







# ----------------------------


#List Hyperparameters that we want to tune.
leaf_size = [1,2]
n_neighbors = [30,35,40,45,50]
p=[2]
#Convert to dictionary
hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
#Create new KNN object
knn_2 = KNeighborsClassifier()
#Use GridSearch
clf = GridSearchCV(knn_2, hyperparameters, cv=10, verbose=3)
#Fit the model
best_model = clf.fit(x_train_tfidf, y_train)
#Print The value of best Hyperparameters
print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
print('Best p:', best_model.best_estimator_.get_params()['p'])
print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])

# Best leaf_size: 1
# Best p: 2
# Best n_neighbors: 40

best_model.best_score_

# ---------------
model = KNeighborsClassifier(leaf_size= 1,n_neighbors=40, p=2)

model.fit(x_train_tfidf, y_train)

y_pred = model.predict(x_test_tfidf)


print(confusion_matrix(y_pred, y_test))

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# [[9974 2679 2432]
#  [1686 5510 1056]
#  [ 908  691 2992]]
# 0.6615582927527929
#               precision    recall  f1-score   support

#          0.0       0.66      0.79      0.72     12568
#          1.0       0.67      0.62      0.64      8880
#          2.0       0.65      0.46      0.54      6480

#     accuracy                           0.66     27928
#    macro avg       0.66      0.63      0.64     27928
# weighted avg       0.66      0.66      0.65     27928




import pickle

filename = 'G:\GitHub\FINAL_PROJECT\Political-Bias-NLP\KNN_66_vectorized_text.sav'
pickle.dump(model, open(filename, 'wb'))
 
# some time later...
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)


















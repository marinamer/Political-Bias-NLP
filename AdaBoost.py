import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report



# -------------------- LOADING AND TREATING DATA -------------
manifesto_project = pd.read_csv("G:\GitHub\FINAL_PROJECT\Political-Bias-NLP\manifesto_clean.csv")

manifesto_project.dtypes

manifesto_project['processed_text'] = manifesto_project['processed_text'].astype(str)


manifesto_project.rename(columns={'Unnamed: 0':'Index'}, inplace=True)
manifesto_project= manifesto_project.set_index('Index')
manifesto_project['cmp_code'] = manifesto_project['cmp_code'].astype('int')
# manifesto_project['label'] = manifesto_project['label'].astype('category')
manifesto_project['domain_name'] = manifesto_project['domain_name'].astype('category')


# # -----------------SETTING TRAIN AND TEST -----------------------

# We only use 40000 rows because otherwise the model fitting takes way too long
X = manifesto_project.iloc[:40000]['processed_text']
Y = manifesto_project.iloc[:40000]['bias_numeric']

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer()

x_train_tfidf = tfidf_vectorizer.fit_transform(x_train) # using fit_transform on train data


# import pickle
# filename = 'tfidftest.pkl'
# pickle.dump(tfidf_vectorizer, open(filename, 'wb'))         

x_test_tfidf = tfidf_vectorizer.transform(x_test)


# --------- MODEL TRAINING ---------
# Importing the necessary modules
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC


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
#Accuracy(5 estimators): 0.684375
#Accuracy(4 estimators): 0.7255

# ---------- Trying some other parameters

# Create adaboost classifer object
abc =AdaBoostClassifier(n_estimators=10, base_estimator=svc,learning_rate=0.1)

# Train Adaboost Classifer
model = abc.fit(x_train_tfidf, y_train)

#Predict the response for test dataset
y_pred = model.predict(x_test_tfidf)


# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


print('Confusion matrix:')

print(confusion_matrix(y_pred, y_test))
print('')
print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


pickle.dump(abc, open('Ada10est81acc.sav', 'wb'))





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
manifesto = pd.read_csv("G:\GitHub\FINAL PROJECT\Political-Bias-NLP\doc_df_with_meta.csv")

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

manifesto['full'] = ""

for text in manifesto['text']:
    doc_id = manifesto['manifesto_id'][text]
    sentence_id = manifesto['id'][text]
    for text in manifesto['text']:
        if manifesto['manifesto_id'] == doc_id and manifesto['id'] != :
            
            








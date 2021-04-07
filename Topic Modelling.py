import pandas as pd
import numpy as np
import re
import nltk

import pickle
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from pprint import pprint


# -------------------- LOADING DATA -------------
manifesto_project = pd.read_csv("G:\GitHub\FINAL PROJECT\Political-Bias-NLP\manifesto_clean.csv")

manifesto_project.dtypes

manifesto_project['processed_text'] = manifesto_project['processed_text'].astype(str)


manifesto_project.rename(columns={'Unnamed: 0':'Index'}, inplace=True)
manifesto_project= manifesto_project.set_index('Index')
manifesto_project['cmp_code'] = manifesto_project['cmp_code'].astype('int')
manifesto_project['domain_name'] = manifesto_project['domain_name'].astype('category')


manifesto_project.dtypes

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk
nltk.download('wordnet')


stemmer = SnowballStemmer("english")

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


processed_text = manifesto_project['text'].map(preprocess)
processed_text[:10]


dictionary = gensim.corpora.Dictionary(processed_text)
count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break


dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_text]
bow_corpus[4310]


bow_doc_4310 = bow_corpus[4310]
for i in range(len(bow_doc_4310)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_4310[i][0], 
                                               dictionary[bow_doc_4310[i][0]], 
bow_doc_4310[i][1]))

lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=5, id2word=dictionary, passes=2, workers=2)

for idx, topic in lda_model.print_topics(num_words=3):
    print('Topic: {} \nWords: {}'.format(idx, topic))


















































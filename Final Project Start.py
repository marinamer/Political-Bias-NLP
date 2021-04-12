import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords #stopwords library
from nltk.stem import WordNetLemmatizer #lemmatizing library
from nltk.tokenize import word_tokenize #tokenizing library
from nltk.stem.porter import *
from nltk import FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pickle
import sklearn

from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *


nltk.download('stopwords') #stopwords
nltk.download('wordnet') #database of English language
nltk.download('punkt') #tokenization
nltk.download('vader_lexicon')

# ------------ LOAD FILES 

manifesto_project = pd.read_csv("G:\GitHub\FINAL PROJECT\Political-Bias-NLP\Manifesto_Project_processed.csv")

news_df = pd.read_csv("G:\GitHub\FINAL_PROJECT\Political-Bias-NLP\complete_news_df.csv")

with open('Random_Forest_400estimators', 'rb') as file:
    random_forest_400 = pickle.load(file)

news_df.text[2]

# manifesto_project.dtypes


news_df.dtypes


manifesto_project.rename(columns={'Unnamed: 0':'Index'}, inplace=True)
manifesto_project= manifesto_project.set_index('Index')
manifesto_project['cmp_code'] = manifesto_project['cmp_code'].astype('category')
manifesto_project['label'] = manifesto_project['label'].astype('category')
manifesto_project['domain_name'] = manifesto_project['domain_name'].astype('category')
manifesto_project['variable_name'] = manifesto_project['variable_name'].astype('category')
manifesto_project['domain_code'] = manifesto_project['domain_code'].astype('category')




def preprocess(text):
    #strip 
    text = re.sub(r'<b>.+?</b>', '', text) 
    text = re.sub(r'<i>.+?</i>', '', text)
    text = re.sub(r'<.+?>', '', text) # remove all other html tags
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
    
    ##  remove punctuations, non-alphanumeric characters and underscores
    text = re.sub(r'[^\w\s]|\d|_', ' ', text)
    
    text = str(text).lower().strip()
    
    #tokenize
    tokens = word_tokenize(text)
    
    #remove stopwords
    stop_words = stopwords.words('english')
    tokens = [t for t in tokens if t not in stop_words]
    
    #lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    text = " ".join(tokens)
    text = str(text).lower().strip()
    
    return text





















import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import nltk

nltk.download('stopwords') #stopwords
nltk.download('wordnet') #database of English language
nltk.download('punkt') #tokenization
nltk.download('vader_lexicon')    
    


manifesto_project = pd.read_csv("G:\GitHub\FINAL_PROJECT\Political-Bias-NLP\manifesto_clean.csv")

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

# # --------------------TRAINING TFIDF--------------------


X = manifesto_project.iloc[:40000]['processed_text']
tfidfconverter = TfidfVectorizer()

tfidf = tfidfconverter.fit_transform(X)

               
                    
filename = 'tfidfGood.pkl'
pickle.dump(tfidfconverter, open(filename, 'wb'))                    
                    
                    































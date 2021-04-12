# ------------------  this file would look like this:
from nltk.corpus import stopwords
from nltk.tokenize import PunktSentenceTokenizer
from nltk.stem import WordNetLemmatizer 
import nltk
import pickle
import re

nltk.download('stopwords') #stopwords
nltk.download('wordnet') #database of English language
nltk.download('punkt') #tokenization
nltk.download('vader_lexicon')    
    
    
def preprocess(text):
    text = str(text)
    #strip 
    text = re.sub(r'<b>.+?</b>', '', text) 
    text = re.sub(r'<i>.+?</i>', '', text)
    text = re.sub(r'<.+?>', '', text) # remove all other html tags
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
    
    ##  remove punctuations, non-alphanumeric characters and underscores
    text = re.sub(r'[^\w\s]|\d|_', ' ', text)
    
    text = str(text).lower().strip()
    
    #tokenize
    tokenizer = PunktSentenceTokenizer()
    tokens = tokenizer.tokenize(text)
    
    #remove stopwords
    stop_words = stopwords.words('english')
    tokens = [t for t in tokens if t not in stop_words]
    
    #lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    text = " ".join(tokens)
    text = str(text).lower().strip()
    text = [text]
    
    return text



def model(text):
    # Preprocess text
    text = preprocess(text)
    
    # Load TFIDF
    tfidf = pickle.load(open("tfidftest.pkl", "rb" ) )
    text_vectorized = tfidf.transform(text)
    
    # Apply Trained Model   
        
    model = pickle.load(open('Ada10est81acc.sav', 'rb'))
        
    result = model.predict(text_vectorized)
    
    return result





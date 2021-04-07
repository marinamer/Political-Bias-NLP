import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import pickle
from nltk.corpus import stopwords
from nltk.tokenize import PunktSentenceTokenizer
import nltk
import re

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

# # ----------------------------------------


X = manifesto_project.iloc[:40000]['processed_text']
tfidfconverter = TfidfVectorizer()



tfidf = tfidfconverter.fit_transform(X)


                    
                    
filename = 'tfidfGood.pkl'
pickle.dump(tfidfconverter, open(filename, 'wb'))                    
                    
                    

#load the content
tfidf = pickle.load(open("tfidf.pkl", "rb" ) )

print(manifesto_project.iloc[2]['processed_text'])



text = 'Eric Talbot Jensen , Brigham Young University Law School Abstract\nThe year 2020 marks the twentieth anniversary of the passage of United Nations Security Council Resolution (“UNSCR”) 1325, the most important moment in the United Nations’ efforts to achieve world peace through gender equality. Over the past several decades, the international community has strengthened its focus on gender, including the relationship between gender and international peace and security. National governments and the United Nations have taken historic steps to elevate the role of women in governance and peacebuilding. The passage of UNSCR 1325 in 2000 foreshadowed what many hoped would be a transformational shift in international law and politics. However, the promise of gender equality has gone largely unrealized, despite the uncontroverted connection between treatment of women and the peacefulness of a nation.\nThis Article argues for the first time that to achieve international peace and security through gender equality, the United Nations Security Council should transition its approach from making recommendations and suggestions to issuing mandatory requirements under Chapter VII of the U.N. Charter. If the Security Council and the international community believe gender equality is the best indicator of sustainable peace, then the Security Council could make a finding under Article 39 with respect to ‘a threat to the peace’—States who continue to mistreat women and girls pose a threat to international peace and security. Such a finding would trigger the Security Council’s mandatory authority to direct States to take specific actions. In exercising its mandatory authority, the Security Council should organize, support, and train grassroots organizations and require States to do the same. It should further require States to produce a reviewable National Action Plan, detailing how each State will implement its responsibilities to achieve gender equality. The Security Council should also provide culturally sensitive oversight on domestic laws which may act as a restraint on true gender equality. Elizabeth Griffiths, Sara Jarman & Eric T. Jensen, World Peace and Gender Equality: Addressing UN Security Council Resolution 1325’s Weaknesses , 27 M ich. J. G ender & L. 247 (2021). //repository.law.umich.edu/mjgl/vol27/iss2/2'


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

text = preprocess(text)


new_text = tfidfconverter.transform(text)

new_text.shape



tfidf.shape




























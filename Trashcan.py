manifesto_project = pd.read_excel("G:\GitHub\FINAL PROJECT\Political-Bias-NLP\documents_MPDataset_MPDS2020b.xlsx")
anotated = pd.read_csv(r"G:\GitHub\FINAL PROJECT\Political-Bias-NLP\anotated.csv")
codebook = pd.read_csv("G:\GitHub\FINAL PROJECT\Political-Bias-NLP\codebook_MP.csv")
doc_df_with_meta = pd.read_csv(r"G:\GitHub\FINAL PROJECT\Political-Bias-NLP\doc_df_with_meta.csv")



# ---------------- MANIFESTO PROJECT PROCESSING DOCUMENTS ----------

codebook.drop(columns='Unnamed: 0', inplace=True)

codebook.rename(columns={"code": "cmp_code"}, inplace=True)



# with open("G:\GitHub\FINAL PROJECT\listfile.data", 'rb') as filehandle:
#     # read the data as binary data stream
#     news_list = pickle.load(filehandle)

# doc_df_with_meta.dtypes
# codebook.dtypes


# I need to convert the column to numeric 
doc_df_with_meta['cmp_code']=pd.to_numeric(doc_df_with_meta['cmp_code'], errors='coerce')

docs_v2 = doc_df_with_meta.merge(codebook, on='cmp_code', validate='m:1')

docs_v2.columns

docs_v2.drop(columns=['Unnamed: 0', 'eu_code', 'has_eu_code', 'is_primary_doc',
       'may_contradict_core_dataset', 'annotations', 'is_copy_of', 'type','md5sum_text', 'url_original',
       'md5sum_original' , 'handbook' 
       ], inplace=True)

docs_v2 = docs_v2[['title_x','pos','text', 'cmp_code', 'domain_code', 'domain_name',
       'variable_name','description_md', 'label', 'manifesto_id', 'party', 'date', 'language',
       'source', 'id', 'title_y']]

docs_v3 = docs_v2.drop(docs_v2[docs_v2['cmp_code']==0].index)



# rile_index = {left: 103, 105, 106, 107, 202, 403, 404, 406, 412, 413, 504, 506, 701}, 
#              {right: 104, 201, 203, 305, 401, 402, 407, 414, 505, 601, 603, 605, 606}





from tqdm import tqdm # cool library which can save your sanity
tqdm.pandas()
docs_v3['processed_text'] = docs_v3['text'].progress_apply(lambda x: preprocess(x))

docs_v3.drop(columns = ['title_x', 'manifesto_id', 'party', 'language', 'source', 'date', 'id', 'title_y' ], inplace=True)



docs_v3['cmp_code'] = docs_v3['cmp_code'].astype('category')

analyzer = SentimentIntensityAnalyzer()

docs_v3['compound'] = [analyzer.polarity_scores(x)['compound'] for x in docs_v3['text']]
docs_v3['neg'] = [analyzer.polarity_scores(x)['neg'] for x in docs_v3['text']]
docs_v3['neu'] = [analyzer.polarity_scores(x)['neu'] for x in docs_v3['text']]
docs_v3['pos'] = [analyzer.polarity_scores(x)['pos'] for x in docs_v3['text']]

docs_v3['domain_code'] = docs_v3['domain_code'].astype('category')


docs_v3['label'] = docs_v3['label'].astype('category')

docs_v3.dtypes

docs_v3 = docs_v3[['text','processed_text', 
                   'cmp_code', 'label', 
                   'domain_name', 'compound', 
                   'neg', 'neu', 'pos', 
                   'domain_code', 'variable_name', 'description_md']]

docs_v3.to_csv(r'G:\GitHub\FINAL PROJECT\Political-Bias-NLP\Manifesto_Project_processed.csv')


# -------------- PROCESSING NEWS_DF -----------

# news_df['text_processed'] = [preprocess(x)[0] for x in news_df['text']]

# news_df['text_processed_textreturn'] = [preprocess(x)[1] for x in news_df['text']]


# Can't do a bag of words with the tokenized text, as it's a string.
# word_bag = []
# for value in news_df['text_processed_textreturn']:
#     for word in value:
#         word_bag.append(g)


# fdist = FreqDist(word_bag)


# print(fdist)

# top_5k = fdist.most_common(5000)
# word_features = list(fdist.keys())[5000:]

# news_df.drop(columns = ['Unnamed: 0', 'ord_in_thread', 'parent_url',
#                         'highlightText', 'highlightTitle','highlightThreadTitle', 
#                         'language', 'external_links', 'external_images',
#                         'rating', 'crawled'], inplace = True)

# for item in news_list_df.posts:
#     temp = pd.DataFrame.from_dict(item)
#     news_df = news_df.append(temp)

# news_df.set_index('uuid', inplace=True)

# news_df.to_csv(r'G:\GitHub\FINAL PROJECT\complete_news_df.csv')

# print(new_df.uuid.nunique())

# print(new_df.uuid.count())

# print(len(pd.unique(new_df.uuid)))


# ------------- NEWS SENTIMENT ANALYSIS -----------------


# analyzer = SentimentIntensityAnalyzer()

# news_df['compound'] = [analyzer.polarity_scores(x)['compound'] for x in news_df['text']]
# news_df['neg'] = [analyzer.polarity_scores(x)['neg'] for x in news_df['text']]
# news_df['neu'] = [analyzer.polarity_scores(x)['neu'] for x in news_df['text']]
# news_df['pos'] = [analyzer.polarity_scores(x)['pos'] for x in news_df['text']]



# ----------- DATABASE ----------------------



# print(codebook['description_md'].iloc[0])       

docs_v3.dtypes


docs_v3['text'] = docs_v3['text'].astype('unicode')
# print(news_df.columns)

from tqdm import tqdm # cool library which can save your sanity
tqdm.pandas()
docs_v3['processed_text'] = docs_v3['text'].progress_apply(lambda x: preprocess(x))

docs_v3.drop(columns = ['title_x', 'manifesto_id', 'party', 'language', 'source', 'date', 'id', 'title_y' ], inplace=True)



docs_v3['categories'] = docs_v3['cmp_code'].astype('category')


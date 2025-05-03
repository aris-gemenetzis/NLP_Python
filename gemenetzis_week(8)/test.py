from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.casual import casual_tokenize
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np


pd.options.display.width = 120
brown = pd.read_csv('brown.csv')
index = ['d:{}_p:{}_s:{}'.format(brown.iloc[i, 0], brown.iloc[i, 1], brown.iloc[i, 2]) for i in range(len(brown))]
brown = pd.DataFrame(brown.values, columns=brown.columns, index=index)
print(len(brown))
print(brown.head(6))
print(brown.tokenized_text)
dfs = dict(tuple(brown.groupby('filename')))
# for d in dfs:
# print(d, dfs.get(d))
# tfidf = TfidfVectorizer(tokenizer=casual_tokenize, dtype=np.float32)
# tfidf_docs = tfidf.fit_transform(raw_documents=brown.tokenized_text)
# print(len(tfidf.vocabulary_))

# brown = brown[:20]
tfidf = TfidfVectorizer(tokenizer=casual_tokenize, dtype=np.float32)
tfidf_docs = tfidf.fit_transform(raw_documents=brown.tokenized_text)
print(len(tfidf.vocabulary_))
tfidf_docs = tfidf_docs.astype(np.float32).toarray().astype(np.float16)  # too big!!
# memory allocation error. data is too big for toarray() function
tfidf_docs = pd.DataFrame(tfidf_docs)
tfidf_docs = tfidf_docs - tfidf_docs.mean()
print(tfidf_docs.shape)

pca = PCA(n_components=15)
pca = pca.fit(tfidf_docs)
pca_topic_vectors = pca.transform(tfidf_docs)
columns = ['topic{}'.format(i) for i in range(pca.n_components)]
pca_topic_vectors = pd.DataFrame(pca_topic_vectors, columns=columns, index=index)
print(pca_topic_vectors.round(3).head(6))

print(tfidf.vocabulary_)
column_nums, terms = zip(*sorted(zip(tfidf.vocabulary_.values(), tfidf.vocabulary_.keys())))
print(terms)
weights = pd.DataFrame(pca.components_, columns=terms, index=['topic{}'.format(i) for i in range(16)])
pd.options.display.max_columns = 8
print(weights.head(4).round(3))

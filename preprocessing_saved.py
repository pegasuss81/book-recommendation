import pandas as pd
import numpy as np

from sklearn import base
from sklearn.feature_extraction import DictVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix
import scipy.sparse

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from spacy.lang.en.stop_words import STOP_WORDS
import spacy

import re
import os
import dill

# Dataframe for every review
df_merge_new_in = pd.read_csv('data/df_merge_with_URL_1.csv')

# Dataframe sorted for each book/title
df_merge_review_URL = pd.read_csv('data/df_merge_review_title_with_URL_1.csv')

def preprocess_collab():
    chunk_size = 5000
    chunks = [x for x in range(0, df_merge_new_in.shape[0], chunk_size)]

    df_merge_pivot = pd.concat([df_merge_new_in.iloc[chunks[i]:chunks[i + 1] - 1].pivot_table(index='title',
                                                                                              columns='reviewerID',
                                                                                              values='overall') for i in
                                range(0, len(chunks) - 1)])

    df_merge_pivot.fillna(0, inplace=True)
        
    # trained model for collaborative filtering
    with open("data/model_knn_1.dill", "rb") as f:
        model_knn = dill.load(f)
    return df_merge_pivot, model_knn
    
# Prepare data as a dictionary that can be fed into DictVectorizer
class DictEncoder(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self, col):
        self.col = col
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        
        def to_dict(l):
            try:
                return {x: 1 for x in l}
            except TypeError:
                return {}
        
        return X[self.col].apply(to_dict)
    
def preprocess_vect():
    merge_review_pipe = Pipeline([
        ('encoder', DictEncoder('reviewText')),
        ('vectorizer', DictVectorizer())
    ])
    merge_desc_pipe = Pipeline([
        ('encoder', DictEncoder('description')),
        ('vectorizer', DictVectorizer())
    ])

    return merge_review_pipe, merge_desc_pipe

def preprocess_wv():
    ## average word vectors for each book
    # with open("data/average_vec.dill", "wb") as f:
    #     dill.dump(average_vec, f)
        
    with open("data/cosine_similarities_1.dill", "rb") as f:
        cosine_similarities = dill.load(f)
    return cosine_similarities


    
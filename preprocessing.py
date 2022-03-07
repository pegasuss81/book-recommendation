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

# Dataframe for every review
df_merge_new_in = pd.read_csv('data/df_merge_with_URL.csv')

# Dataframe sorted for each book/title
df_merge_review_URL = pd.read_csv('data/df_merge_review_title_with_URL.csv')

def preprocess_collab():
    chunk_size = 5000
    chunks = [x for x in range(0, df_merge_new_in.shape[0], chunk_size)]

    df_merge_pivot = pd.concat([df_merge_new_in.iloc[chunks[i]:chunks[i + 1] - 1].pivot_table(index='title',
                                                                                              columns='reviewerID',
                                                                                              values='overall') for i in
                                range(0, len(chunks) - 1)])

    df_merge_pivot.fillna(0, inplace=True)
    df_merge_matrix = csr_matrix(df_merge_pivot)
    df_merge_matrix.todense()
    scipy.sparse.save_npz('data/sparse_matrix_with_URL.npz', df_merge_matrix)

    # df_merge_matrix = csr_matrix(df_merge_pivot)
    sparse_matrix = scipy.sparse.load_npz('data/sparse_matrix_with_URL.npz')

    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    # model_knn = NearestNeighbors(metric='cosine', algorithm='ball_tree')
    model_knn.fit(sparse_matrix)
    return df_merge_matrix, model_knn


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
    w2v_review_each_book = Word2Vec.load('data/combined_review_children_books_w2v.model')

    def vectorize_combined_reviews(data, maxlen=100, embedding_dim=30):
        """
        Returns a 3D array of shape
        n reviews x maxlen x embedding_dim.
        """
        # Create empty array
        vectorized_data = np.zeros(shape=(len(data), maxlen, embedding_dim))

        for row, review in enumerate(data):
            # Preprocess each review
            tokens = simple_preprocess(review)

            # Truncate long reviews
            if len(tokens) > maxlen:
                tokens = tokens[:maxlen]

            # Get vector for each token in review
            for col, token in enumerate(tokens):
                try:
                    word_vector = w2v_review_each_book.wv[token]
                    # Add vector to array
                    vectorized_data[row, col] = word_vector[:embedding_dim]
                except KeyError:
                    pass

        return vectorized_data

    maxlen = 100  # Our predetermined limit
    embedding_dim = 30  # The first 30 values in our w2v vectors

    X = vectorize_combined_reviews(df_merge_review_URL.reviewText, maxlen, embedding_dim)

    print('Shape of feature matrix:', X.shape)

    # calculate mean of vectors for each book
    average_vec = []

    for i in range(len(X)):
        average = np.mean(X[i], axis=0)
        average_vec.append(average)

    return average_vec


def to_dataframe(rec_tuple):
    df = pd.DataFrame(rec_tuple).T
    df.columns = ["title", "distance"]
    return df


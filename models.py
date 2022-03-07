from preprocessing import preprocess_collab, preprocess_vect, preprocess_wv

df_merge_pivot, model_knn = preprocess_collab()
merge_review_pipe, merge_desc_pipe = preprocess_vect()
average_vec = preprocess_wv()

# Recommender from user ratings - collaborative filtering. pivot table + NearestNeighbors
def book_recommender_collab(string):
    title = df_merge_pivot[df_merge_pivot.index.str.contains(string)].index[0]

    distances, indices = model_knn.kneighbors(df_merge_pivot.loc[title, :].values.reshape(1, -1), n_neighbors=21908)
    titles = df_merge_pivot.index[np.array(indices.flatten())]

    return titles, distances.flatten()

# Recommender using reviewText and description for each book. vectorizer + FeatureUnion + NearestNeighbors
def book_recommender_text_features(w1, w2, string):
    """
    book recommendation system using
    w1: weight for review feature
    w2: weight for description feature
    string: substring of a title
    """
    union_merge = FeatureUnion([('reviewText', merge_review_pipe),
                                ('description', merge_desc_pipe)],
                               transformer_weights={
                                   'reviewText': w1,
                                   'description': w2
                               })
    features_merge_review = union_merge.fit_transform(df_merge_review_URL)

    union_merge_review_model = NearestNeighbors(metric='cosine', algorithm='brute')
    union_merge_review_model.fit(features_merge_review)

    index1 = df_merge_review[df_merge_review.title.str.contains(string)].index[0]
    title1 = df_merge_review[df_merge_review.title.str.contains(string)]['title'].values[0]

    # distances, indices = union_merge_review_model.kneighbors(features_merge_review[index1], n_neighbors=6)
    distances, indices = union_merge_review_model.kneighbors(features_merge_review[index1], n_neighbors=df_merge_review_URL.shape[0])
    titles = df_merge_review['title'][df_merge_review.index[np.array(indices.flatten())]].tolist()
    # print(titles)

    # return distances, indices, titles
    return titles, distances.flatten()

# Recommender using Word2Vec model

def book_recommender_wv(string):
    # finding cosine similarity for the vectors
    cosine_similarities = cosine_similarity(average_vec, average_vec)
    # print(type(cosine_similarities))

    # title
    # books = df_merge_review_URL[['title']]
    # print(books)
    # Reverse mapping of the index
    indices = pd.Series(df_merge_review_URL.index, index=df_merge_review_URL['title']).drop_duplicates()
    # print(indices)

    title = df_merge_review_URL[df_merge_review_URL.title.str.contains(string) == True].index[0]
    idx = indices[title]
    # print(title, idx) # title == idx?
    # sim_scores = list(enumerate(cosine_similarities[idx]))
    # sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)
    sim_scores = list(enumerate(1 - cosine_similarities[idx]))
    # print(sim_scores)
    ##sim_scores = sorted(sim_scores, key = lambda x: x[1])
    # print(sim_scores)
    titles = df_merge_review_URL['title'][df_merge_review_URL.index[np.array(indices)]].tolist()
    # sim_scores = sim_scores[1:6]
    return titles, sim_scores
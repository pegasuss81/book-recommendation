
# book-recommendation

This is a book recommendation system in attempt to incorporate user's input for several factors i.e user ratings, text reviews, and book descriptions. I use Amazon review data scrawled from user reviews ranging from 1996 to 2018. 
https://nijianmo.github.io/amazon/index.html#subsets

The whole data consists of ~27M reviews but for my project I am using a subset of data, more specifically the ones categorized as children’s books (400K reviews for about 22K books). To make use of several aspects of data we have in hand and also to give better customized recommendations (and result in better performance) I built 3 different machine learning models and then combined those models giving each weights. These weights will be determined by user inputs reflecting which factors they think are the most and the least relevant to their selection of books. 

# Collaborative filtering (M1)
It is based on users’ historical preference and the underlying assumption is that users who liked similar items in the past will likely to choose the similar items in the future. I used Nearest Neighborhood and it basically calculates the similarities between items and select the top N similar items using cosine similarities. 

# Vectorizer of the book descriptions and text reviews + Feature Union (M2)
This model is to see the similarity of items using vectorizer. Both book description and text reviews are used together with feature union. Weights of each category will be given by users but I find that book descriptions are better in giving better recommendations. Similar book descriptions are more efficient than similar reviews in this method.

# Word2Vec word embedding (M3)
While vectorizer focuses on frequency of words in the corpus, word embedding is good at capturing semantic relevance of words in the content. I use Word2Vec from gensim and calculate the average vectors for each books. Comparing their cosine similarities in the vector space, the top N similar books can be recommended.

# Combined Model
I consolidate these three ML models and calculate the final metric that will require inputs from users,  which will be used as weights for each model. i.e.

```Resulting_Metric = w1 * distance(M1) + w2 * distance(M2) + w3 * distance(M3)```

# Building a flask app

# Deploy the app online with Heroku 

import flask
from flask import render_template, request, redirect
import difflib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from functools import reduce

from preprocessing import to_dataframe
from models import book_recommender_collab, book_recommender_text_features, book_recommender_wv

app = flask.Flask(__name__, template_folder='templates')


def combined_model(w_collab=1.0, w_vect_desc=1.0, w_vect_review=0.2, w_feature_union=1.0, w_wv=1.0, string="Bambi",
                   n_rec=5):
    df_collab = to_dataframe(book_recommender_collab(string))
    df_vect = to_dataframe(book_recommender_text_features(w_vect_desc, w_vect_review, string))
    df_wv = to_dataframe(book_recommender_wv(string))
    df_wv['distance'] = df_wv['distance'].str[1]

    df_join = reduce(lambda left, right: pd.merge(left, right, on=['title'],
                                                  how='outer'), [df_collab, df_vect, df_wv])
    df_join.columns = ['title', 'dist_collab', 'dist_vect', 'dist_wv']

    df_join['dist_metric'] = w_collab * df_join['dist_collab'] + w_feature_union * df_join['dist_vect'] \
                             + w_wv * df_join['dist_wv']

    return df_join.sort_values('dist_metric')[["title", "dist_metric"]].head(5)


# Set up the main route
@app.route('/', methods=['GET'])
def main():
    if flask.request.method == 'GET':
        return (flask.render_template('index.html'))

@app.route('/features_input', methods=['GET'])
def get_user_input():
    title = request.form.get("title")
    first_weight = request.form.get("value1")
    seconde_weight = request.form.get("value2")
    third_weight = request.form.get("value3")

    combined_model(first_weight, 1.0, 0.2, seconde_weight, third_weight, title)
    #print(title, first_weight, seconde_weight, third_weight)
    return flask.render_template('index.html')

# def my_form_post(): #
#     text = request.form['text']
#     processed_text = text.upper()
#     return processed_text
# =============================

if __name__ == '__main__':
    #app.run()
    app.run(debug=True, port=3000)

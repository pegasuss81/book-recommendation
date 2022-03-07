import flask
import difflib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from functools import reduce

#from preprocessing import to_dataframe
#from models import book_recommender_collab, book_recommender_text_features, book_recommender_wv

app = flask.Flask(__name__, template_folder='templates')

# df2 = pd.read_csv('./model/tmdb.csv')
#
# count = CountVectorizer(stop_words='english')
# count_matrix = count.fit_transform(df2['soup'])
#
# cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
#
# df2 = df2.reset_index()
# indices = pd.Series(df2.index, index=df2['title'])
# all_titles = [df2['title'][i] for i in range(len(df2['title']))]


# def combined_model(w_collab=1.0, w_vect_desc=1.0, w_vect_review=0.2, w_feature_union=1.0, w_wv=1.0, string="Bambi",
#                    n_rec=5):
#     df_collab = to_dataframe(book_recommender_collab(string))
#     df_vect = to_dataframe(book_recommender_text_features(w_vect_desc, w_vect_review, string))
#     df_wv = to_dataframe(book_recommender_wv(string))
#     df_wv['distance'] = df_wv['distance'].str[1]
#
#     df_join = reduce(lambda left, right: pd.merge(left, right, on=['title'],
#                                                   how='outer'), [df_collab, df_vect, df_wv])
#     df_join.columns = ['title', 'dist_collab', 'dist_vect', 'dist_wv']
#
#     df_join['dist_metric'] = w_collab * df_join['dist_collab'] + w_feature_union * df_join['dist_vect'] \
#                              + w_wv * df_join['dist_wv']
#
#     return df_join.sort_values('dist_metric')[["title", "dist_metric"]].head(10)


# Set up the main route
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return (flask.render_template('index.html'))

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    processed_text = text.upper()
    return processed_text
    # if flask.request.method == 'POST':
    #     b_name = flask.request.form['title']
    #     b_name = b_name.title()
    #     #        check = difflib.get_close_matches(m_name,all_titles,cutout=0.50,n=1)
    #     if b_name not in all_titles:
    #         return (flask.render_template('negative.html', name=b_name))
    #     else:
    #         result_final = combined_model()
    #
    #         #return flask.render_template('positive.html', movie_names=names, movie_date=dates, search_name=m_name)
    #         #return flask.render_template('positive.html', movie_names=names)
    #         return result_final

if __name__ == '__main__':
    #app.run()
    app.run(debug=True, port=3000)
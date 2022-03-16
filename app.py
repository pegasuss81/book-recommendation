from flask import Flask, render_template, request, redirect
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.metrics.pairwise import cosine_similarity
from functools import reduce
import dill

#from preprocessing_saved import to_dataframe
from models import book_recommender_collab, book_recommender_text_features, book_recommender_wv, to_dataframe

app = Flask(__name__, template_folder='templates')

with open("data/cosine_similarities_1.dill", "rb") as f:
        cosine_similarities = dill.load(f)
        
def combined_model(w_collab=1.0, w_vect_desc=1.0, w_vect_review=0.2, w_feature_union=1.0, w_wv=1.0, string="Bambi",
                   n_rec=5):
    df_collab = to_dataframe(book_recommender_collab(string))
    df_vect = to_dataframe(book_recommender_text_features(w_vect_desc, w_vect_review, string))
    df_wv = to_dataframe(book_recommender_wv(string, cosine_similarities))
    df_wv['distance'] = df_wv['distance'].str[1]

    df_join = reduce(lambda left, right: pd.merge(left, right, on=['title'],
                                                  how='outer'), [df_collab, df_vect, df_wv])
    df_join.columns = ['title', 'dist_collab', 'dist_vect', 'dist_wv']

    df_join['dist_metric'] = w_collab * df_join['dist_collab'] + w_feature_union * df_join['dist_vect'] \
                             + w_wv * df_join['dist_wv']

    #return df_join.sort_values('dist_metric')[["title", "dist_metric", "URL", "image"]].head(5)
    return df_join.sort_values('dist_metric')[["title"]].head(n_rec)

# Set up the main route
@app.route('/', methods=['GET'])
def main():
    if request.method == 'GET':
        return render_template('index.html')

@app.route('/features_input', methods=['GET'])
def get_recommendataions():
    title = request.args.get("title")
    first_weight = request.args.get("features")
    second_weight = request.args.get("features-2")
    third_weight = request.args.get("features-3")
    
    if first_weight == "value1":
        value1 = 1.0
        if second_weight == "value2":
            value2 = 0.5
            value3 = 0.2
        elif second_weight == "value3":
            value3 = 0.5
            value2 = 0.2
        
    elif first_weight == "value2":
        value2 = 1.0
        if second_weight == "value1":
            value1 = 0.5
            value3 = 0.2
        elif second_weight == "value3":
            value3 = 0.5
            value1 = 0.2
   
    elif first_weight == "value3":
        value3 = 1.0
        if second_weight == "value2":
            value2 = 0.5
            value1 = 0.2
        elif second_weight == "value1":
            value1 = 0.5
            value2 = 0.2
    
    recommendations = combined_model(value1, 1.0, 0.2, value2, value3, title)
    
    return render_template('results.html', results = recommendations)

# =============================

if __name__ == '__main__':
    app.run(debug=True)
    #app.run(debug=True, port=3000)

import os
import numpy as np
import pandas as pd
from flask import Flask, flash, request, redirect, url_for, render_template
# from werkzeug.utils import secure_filename




# products_prob = pd.read_csv("products_prob.csv")
item_similarity_matrix = np.load('item_similarity_matrix.npy')
item_user_matrix = pd.read_csv('item_user_matrix.csv', index_col='Description')


def find_similar_items(target_item, item_similarity_matrix, k=5):
    target_item_index = item_user_matrix.index.get_loc(target_item)
    similar_items_indices = item_similarity_matrix[target_item_index].argsort()[::-1][1:k+1]
    similar_items = [(item_user_matrix.index[i], item_similarity_matrix[target_item_index, i]) for i in similar_items_indices]
    return similar_items
####################################################################
app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
@app.route('/index/index', methods=['GET', 'POST'])
def home():
    return render_template("./index.html")

@app.route('/index/<product>', methods=['GET', 'POST'])
def predict(product):
    # l=recommend(product, 3)
    target_item = product
    l = find_similar_items(target_item, item_similarity_matrix, k=3)
    return render_template("./index.html", ob1=l[0][0], ob2=l[1][0], ob3=l[2][0])

@app.route('/about')
@app.route('/index/about')
def about():
    return render_template("./about.html")


if __name__=='__main__':
    app.run()
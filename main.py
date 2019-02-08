#!/usr/bin/python3
from flask import Flask, render_template, request,jsonify, abort
app = Flask(__name__)
from classify import Classifier
from document import Document
from newspaper import Article
# import nltk
import sys

c = Classifier()
# database = open_database()
# nltk.download('wordnet')

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/check', methods=['POST'])
def check():
    print (request.form, file=sys.stderr)
    if not request.form or not 'link' in request.form:
        abort(400)
    try:
        url = request.form['link']
        article = Article(url)
        article.download()
        article.parse()

        d = Document(article.title, article.text)
        result = d.compute_val(c)

        if result == 1:
            return render_template('success.html')
        else:
            return render_template('failure.html')
    except:
        return render_template('error.html')
    # hashed_val = hash(d)
    # if hashed_val in database:
    #     return database.get(hashed_val)
    # else:
    #     sens = d.compute_val(c)
    #     database.write(hashed_val, sens)
    #     return sens

#!/usr/bin/python3
from flask import Flask, render_template, request,jsonify, abort
app = Flask(__name__)
from classify import Classifier
# from document import Document
from newspaper import Article
import sys

c = Classifier()
# database = open_database()
app.config['TEMPLATES_AUTO_RELOAD'] = True

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

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

        # d = Document(article.title, article.text)
        result = c.classify(article.title, article.text)

        if result > 0.6:
            return render_template('success.html', percentage=round(result*100,3))
        else:
            return render_template('failure.html', percentage=round(result*100,3))
    except:
        return render_template('error.html')
    # hashed_val = hash(d)
    # if hashed_val in database:
    #     return database.get(hashed_val)
    # else:
    #     sens = d.compute_val(c)
    #     database.write(hashed_val, sens)
    #     return sens

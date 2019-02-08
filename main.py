#!/usr/bin/python3
from flask import Flask, render_template, request,jsonify, abort
app = Flask(__name__)
from classify import Classifier
from document import Document
import sys

c = Classifier()
# database = open_database()

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/check', methods=['POST'])
def check():
    print (request.form, file=sys.stderr)
    if not request.form or not 'title' in request.form:
        abort(400)
    d = Document(request.form['title'], request.form['text'])
    result = d.compute_val(c)
    title = request.form['title']
    if result == 1:
        return render_template('success.html', value=title)
    else:
        return render_template('failure.html', value=title)
    # hashed_val = hash(d)
    # if hashed_val in database:
    #     return database.get(hashed_val)
    # else:
    #     sens = d.compute_val(c)
    #     database.write(hashed_val, sens)
    #     return sens

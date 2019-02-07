#!/usr/bin/python3
from flask import Flask, render_template, request,jsonify
app = Flask(__name__)
from classify import Classifier
from document import Document

c = Classifier()
# database = open_database()

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/check', methods=['POST'])
def check():
    if not request.json or not 'title' in request.json:
        abort(400)
    d = Document(request.json['title'], request.json['text'])
    return jsonify({'sensationalist':d.compute_val(c)}), 201
    hashed_val = hash(d)
    # if hashed_val in database:
    #     return database.get(hashed_val)
    # else:
    #     sens = d.compute_val(c)
    #     database.write(hashed_val, sens)
    #     return sens

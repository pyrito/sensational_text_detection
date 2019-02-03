from flask import Flask, render_template
app = Flask(__name__)
from classify import Classifier
from document import Document

c = Classifier()
database = open_database()

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/check')
def check(title, text):
    d = Document(title, text)
    hashed_val = hash(d)
    if hashed_val in database:
        return database.get(hashed_val)
    else:
        sens = d.compute_val(c)
        database.write(hashed_val, sens)
        return sens

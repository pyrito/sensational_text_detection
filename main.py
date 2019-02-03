from flask import Flask
app = Flask(__name__)
from classify import Classifier
from document import Document

c = Classifier()
database = open_database()

@app.route('/')
def hello_world():
    return 'Hello, testing!'

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

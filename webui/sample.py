# A sample web UI for ArtNet.

from flask import Flask, render_template

app = Flask(__name__)


@app.route('/hello_world')
def hello_world():
    return "Hello world!"


@app.route('/')
def index():
    return render_template('sample.html')

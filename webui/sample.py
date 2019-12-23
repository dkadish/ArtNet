"""A sample web UI for ArtNet."""

from flask import Flask, render_template, request, redirect
from model import ArtNetModel

app = Flask(__name__)


@app.route('/hello_world')
def hello_world():
    """Just a test that Flask is up and running."""
    return "Hello world!"


@app.route('/')
def index():
    """Index route."""
    return render_template('sample.html')


@app.route('/upload_image', methods=['GET', 'POST'])
def upload_image():
    """Images get uploaded to this route."""
    if request.method == 'POST':
        image = request.files['image']
        blob = image.read()
        app.logger.debug('Received %s, %d bytes', image, len(blob))

        model = ArtNetModel()
        app.logger.debug('Loaded model %s', model)

        app.logger.debug('Predicting something')
        pred = model.predict(blob)
        app.logger.debug('Predicted', [v.shape for v in pred])

        return redirect(request.url)

    return render_template('sample.html')

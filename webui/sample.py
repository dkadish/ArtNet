"""A sample web UI for ArtNet."""

import logging
from flask import Flask, render_template, request, redirect
from flask.logging import default_handler
from model import ArtNetModel

app = Flask(__name__)

# Get the root logger, and channel the app.log there.
# logging.basicConfig(filename="logs/sample.log")
root_logger = logging.getLogger()
root_logger.addHandler(default_handler)

@app.route('/hello_world')
def hello_world():
    """Just a test that Flask is up and running."""
    return "Hello world and everyone!"


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
        bboxes, probs, width, height = model.predict(blob)
        app.logger.debug('Predicted', bboxes, probs)

        # return redirect(request.url)
        return render_template('results.html',
                               bboxes=bboxes, probs=probs,
                               width=width, height=height)

    return render_template('sample.html')

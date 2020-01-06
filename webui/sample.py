"""A sample web UI for ArtNet."""

import logging
import base64
from flask import Flask, render_template, request, redirect
from flask.logging import default_handler
from model import ArtNetModel

app = Flask(__name__)

# Get the root logger, and channel the app.log there.
# logging.basicConfig(filename="logs/sample.log")
root_logger = logging.getLogger()
root_logger.addHandler(default_handler)
root_logger.setLevel(logging.DEBUG)

@app.route('/hello_world')
def hello_world():
    """Just a test that Flask is up and running."""
    app.logger.debug("Route /hello_world")
    return "Hello world and everyone!"


@app.route('/')
def index():
    """Index route."""
    app.logger.debug("Route /")
    return render_template('sample.html')


@app.route('/upload_image', methods=['GET', 'POST'])
def upload_image():
    """Images get uploaded to this route."""
    app.logger.debug("Route /upload_image")
    if request.method == 'POST':
        image = request.files['image']
        blob = image.read()
        app.logger.debug('Received %s, %d bytes', image, len(blob))

        model = ArtNetModel()
        app.logger.debug('Loaded model %s', model)

        app.logger.debug('Predicting something')
        bboxes, probs, width, height = model.predict(blob)
        app.logger.debug('Predicted', bboxes, probs)

        # Or image.seek(0) and then image
        # image_echo = base64.b64encode(image.stream.read()).decode('ascii')
        image_echo = base64.b64encode(blob).decode('ASCII')
        app.logger.debug("Echoing %d bytes of base64 encoded stuff", len(image_echo))

        # return redirect(request.url)
        return render_template('results.html',
                               image=image_echo,
                               bboxes=bboxes, probs=probs,
                               width=width, height=height)

    return render_template('sample.html')

@app.route('/upload_image_predict_onto', methods=['GET', 'POST'])
def upload_image_image_predict_onto():
    """Images get uploaded to this route."""
    app.logger.debug("Route /upload_image")
    if request.method == 'POST':
        image = request.files['image']
        blob = image.read()
        app.logger.debug('Received %s, %d bytes', image, len(blob))

        model = ArtNetModel()
        app.logger.debug('Loaded model %s', model)

        app.logger.debug('Predicting something')
        prediction_bboxes = model.predict_onto_image(blob)
        app.logger.debug('Predicted something')

        # Or image.seek(0) and then image
        # image_echo = base64.b64encode(image.stream.read()).decode('ascii')
        image_bboxes = base64.b64encode(
            # prediction_bboxes).decode('ASCII')
            prediction_bboxes).decode('ASCII')
        app.logger.debug(
            "Echoing %d bytes of base64 encoded stuff. Beginning looks like %s",
            len(image_bboxes),
            image_bboxes[:100]
        )

        # return redirect(request.url)
        return render_template('results_onto.html',
                               image=image_bboxes)

    return render_template('sample.html')

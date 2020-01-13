"""A sample web UI for ArtNet."""

import logging
import base64
from io import BytesIO
from flask import Flask, render_template, request, redirect
from flask.logging import default_handler
from PIL import Image
from artnetmodel import ArtNetModel

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
        payload = request.files['image']
        image = Image.open(payload)

        # blob = ui.BytesIO(image).getvalue()
        # blob = payload.read()
        # image = Image.open(blob)
        app.logger.debug("Received %s", image)

        model = ArtNetModel('results/models/model-10000.pth')
        app.logger.debug('Loaded model %s', model)

        app.logger.debug('Predicting something')
        bboxes, classes, probs = model.predict(image)
        app.logger.debug('Predicted', bboxes, classes, probs)

        width, height = image.size

        # Echo the image back to the user
        buf = BytesIO()
        image.save(buf, format="JPEG")
        payload_echo = base64.b64encode(buf.getvalue()).decode('ASCII')
        app.logger.debug("Echoing %d bytes of base64 encoded stuff", len(payload_echo))

        # return redirect(request.url)
        return render_template('results.html',
                               image=payload_echo,
                               bboxes=bboxes.tolist(),
                               classes=classes.tolist(),
                               probs=probs.tolist(),
                               width=width, height=height)

    return render_template('sample.html')

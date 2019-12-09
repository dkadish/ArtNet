# A sample web UI for ArtNet.

from flask import Flask, render_template, request, redirect

app = Flask(__name__)


@app.route('/hello_world')
def hello_world():
    return "Hello world!"


@app.route('/')
def index():
    return render_template('sample.html')


@app.route('/upload_image', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        image = request.files['image']
        print(f"{image}, {image.content_length} bytes")
        return redirect(request.url)

    return render_template('sample.html')

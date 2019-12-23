# ArtNet

ArtNet dataset construction project at ITU

Due to some Flask interaction something something, requires TensorFlow < 1.14, see [ModuleNotFoundError: No module named 'tensorflow_core.keras' in Flask](https://github.com/tensorflow/tensorflow/issues/34607)

# Web interface

When starting from command line, rather than WSGI etc, in directory ArtNet say

    source bin/env/activate
    export FLASK_APP="webui.sample"
    # export FLASK_DEBUG=1
    # export FLASK_ENV="development"
    # flask run
    python3 -m flask run # not `flask run` because it might run system wide flask

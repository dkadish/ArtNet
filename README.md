# ArtNet

ArtNet dataset construction project at ITU. This is an object recognition model based on [*pottershu*'s Python implementation of Faster-RCNN](https://github.com/potterhsu/easy-faster-rcnn.pytorch), and trained on open data from [National Gallery of Denmark](https://smk.dk).

# Web interface

When starting from command line, rather than WSGI etc, in directory ArtNet say

    source bin/env/activate
    export FLASK_APP="webui.sample"
    # export FLASK_DEBUG=1
    # export FLASK_ENV="development"
    # flask run
    python3 -m flask run # not `flask run` because it might run system wide flask

![](webui-with-slider.gif)

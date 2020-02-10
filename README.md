# ArtNet

ArtNet dataset construction project at ITU. This is an object recognition model based on [*pottershu*'s Python implementation of Faster-RCNN](https://github.com/potterhsu/easy-faster-rcnn.pytorch), and trained on open data from [National Gallery of Denmark](https://smk.dk).

# Data and sample sets

SMK Open Data artwork metadata was downloaded as a JSON.

A first sample set of 1000 images, stratified as 250 images of *blyant*, *maleri*, *pen* and *akvarel* each and called `bam-stratified-sample1000`, was sent to Amazon Mechanical Turk for turkers to classify whether the images contain a person or people, or don't. Of these, the images which had top annotator agreement (3/3) Mace drew bounding boxes using [labelImg](https://github.com/tzutalin/labelImg). These bounding boxes were used to train the object recognition model.

A second sample set called `sample-next5000` was produced, but processing it on AMT caused server issues and was aborted.

Finally, the rest of the SMK images were classified with the same task on AMT, excluding the images which were hosted by the `iip.smk.dk` server which was causing troubles earlier. This sample set is called `the-rest-after-next5000-not-on-iip-smk-dk` and covers about 10000 images.

On ownCloud, the `amt_results` directory contains the results of work the turkers performed. Bounding boxes are stored on ownCloud too.

# Model

The trained models are stored on ownCloud.

# Web interface

A sample web interface is running at https://modgift.itu.dk/artnet/. It prompts an image upload from the user, and returns the image with predicted bounding boxes overlaid on it.

When starting the web interface from command line, rather than WSGI etc, in directory ArtNet say the following:

    source bin/env/activate
    export FLASK_APP="webui.sample"
    # export FLASK_DEBUG=1
    # export FLASK_ENV="development"
    # flask run
    python3 -m flask run # not `flask run` because it might run system wide flask

![](webui-with-slider.gif)

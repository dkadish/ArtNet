"""Model for ArtNet."""

import sys
import logging
import pickle
import numpy as np
import cv2
sys.path.insert(0, '../Keras-FasterRCNN')
# from tensorflow import keras
import keras_frcnn.resnet as nn
from keras_frcnn import config
from keras import backend as K
from keras.layers import Input
from keras.models import Model


class ArtNetModel():
    """Model for ArtNet."""

    def __init__(self, config_file='model/config.pickle'):
        """Initialize the class."""
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)

        with open(config_file, 'rb') as fd:
            self.C = pickle.load(fd)

        print(self.C)
        self.rpn, self.classifier, self.classifier_only = self.build_models()

    def build_models(self):
        """Build models."""
        class_mapping = {v: k for k, v in self.C.class_mapping.items()}
        img_input = Input(shape=(None, None, 3))
        roi_input = Input(shape=(self.C.num_rois, 4))
        feature_map_input = Input(shape=(None, None, 1024))

        # Define base network
        shared_layers = nn.nn_base(img_input, trainable=True)

        # Define RPN
        num_anchors = len(self.C.anchor_box_scales) * len(self.C.anchor_box_ratios)
        rpn_layers = nn.rpn(shared_layers, num_anchors)

        # The classifier
        classifier = nn.classifier(
            feature_map_input, roi_input,
            self.C.num_rois, nb_classes=len(class_mapping), trainable=True)

        # Define the models
        model_rpn = Model(img_input, rpn_layers)
        model_classifier_only = Model([feature_map_input, roi_input], classifier)
        model_classifier = Model([feature_map_input, roi_input], classifier)

        # Load weights
        model_rpn.load_weights(self.C.model_path, by_name=True)
        model_classifier.load_weights(self.C.model_path, by_name=True)

        # Compile the models
        model_rpn.compile(optimizer='sgd', loss='mse')
        model_classifier.compile(optimizer='sgd', loss='mse')

        return (model_rpn, model_classifier, model_classifier_only)

    def predict(self, data):
        """Predict on data received."""
        img = self.construct_image(data)
        # bytes_as_np_array = np.frombuffer(buffer.read(), dtype=np.uint8)
        X, ratio = self.format_img(img, self.C)

        if K.common.image_dim_ordering() == 'tf':
            X = np.transpose(X, (0, 2, 3, 1))

        # Get the feature maps and output from RPN
        [Y1, Y2, F] = self.rpn.predict(X)

        return Y1, Y2, F

    def construct_image(self, data):
        """Construct an image from received data."""
        img = cv2.imdecode(np.asarray(bytearray(data)), cv2.IMREAD_UNCHANGED)
        self.logger.debug('Read image of shape %s', img.shape)

        return img






    

    # The below is just copied from test_frcnn.py, and modified for this class

    def format_img_size(self, img, C):
        """Formats the image size based on configuration C."""
        img_min_side = float(self.C.im_size)
        (height, width, _) = img.shape

        if width <= height:
            ratio = img_min_side/width
            new_height = int(ratio * height)
            new_width = int(img_min_side)
        else:
            ratio = img_min_side/height
            new_width = int(ratio * width)
            new_height = int(img_min_side)

        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        return img, ratio

    def format_img_channels(self, img, C):
        """Formats the image channels based on configuration C."""
        img = img[:, :, (2, 1, 0)]
        img = img.astype(np.float32)
        img[:, :, 0] -= self.C.img_channel_mean[0]
        img[:, :, 1] -= self.C.img_channel_mean[1]
        img[:, :, 2] -= self.C.img_channel_mean[2]
        img /= self.C.img_scaling_factor
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        return img

    def format_img(self, img, C):
        """Formats an image for model prediction based on configuration C."""
        img, ratio = self.format_img_size(img, C)
        img = self.format_img_channels(img, C)

        return img, ratio

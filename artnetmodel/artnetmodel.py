"""Model for ArtNet."""

import logging
import sys
sys.path.insert(0, '../../easy-faster-frcnn.pytorch')
logging.debug("sys.path now %s", sys.path)
import torch
import glob
import numpy as np
from PIL import ImageDraw, ImageFont
from sparklines import sparklines
from torchvision.transforms import transforms
from dataset.base import Base as DatasetBase
from backbone.base import Base as BackboneBase
from bbox import BBox
from model import Model # from upstream
from roi.pooler import Pooler
from config.eval_config import EvalConfig as Config


class ArtNetModel():
    """Model for ArtNet."""

    def __init__(self, filename):
        """Initialize the class."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.dataset_class = DatasetBase.from_name('voc2007')
        self.backbone = BackboneBase.from_name('resnet101')(pretrained=False)
        self.model = Model(self.backbone,
                           self.dataset_class.num_classes(),
                           pooler_mode=Pooler.Mode.ALIGN,
                           anchor_ratios=[(1, 2), (1, 1), (2, 1)],
                           anchor_sizes=[128, 256, 512],
                           rpn_pre_nms_top_n=6000,
                           rpn_post_nms_top_n=300).cpu()

        self.load(filename)

    def load(self, filename):
        """Load the weights from file."""
        self.logger.debug("Loading from %s", filename)
        checkpoint = torch.load(filename,
                                map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['state_dict'])
        # TODO: Add some check here that loading went as intended

    def predict(self, data):
        """Predict objects from data."""
        with torch.no_grad():
            # image = transforms.Image.open(data)
            image = data
            image_tensor, scale = self.dataset_class.preprocess(
                image,
                Config.IMAGE_MIN_SIDE,
                Config.IMAGE_MAX_SIDE)

            self.logger.debug("Ok predicting from %s", image)
            detection_bboxes, detection_classes, detection_probs, _ = self.model.eval().forward(image_tensor.unsqueeze(dim=0).cpu())
            detection_bboxes /= scale

            self.logger.debug("Bboxes %s", detection_bboxes)
            self.logger.debug("Classes %s", detection_classes)
            self.logger.debug("Probs %s", detection_probs)
            self.logger.debug("Probs distribution %s",
                              sparklines(detection_probs.tolist()))
            probs_freq, probs_bins = np.histogram(detection_probs.tolist(),
                                                  range=(0, 1))
            self.logger.debug("Histogram of probs distribution %d %s %d",
                              probs_bins[0],
                              sparklines(probs_freq),
                              probs_bins[-1]
            )
            self.logger.debug("Pruned Histogram of probs distribution %d %s %d",
                              probs_bins[1],
                              sparklines(probs_freq[1:]),
                              probs_bins[-1]
            )

            # Some bboxes might have nan coordinates; prune them
            # TODO: Is this still so?
            mask = ~np.isnan(detection_bboxes).any(axis=1).bool()
            self.logger.debug("Pruned from %d to %d bboxes",
                              len(detection_bboxes),
                              sum(mask))

            return detection_bboxes[mask], detection_classes[mask], detection_probs[mask]

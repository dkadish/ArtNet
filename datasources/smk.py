"""Statens Museum for Kunst."""

import logging
import pandas as pd
from urllib.parse import urlparse

SEED = 2019
DATAFILE = 'data/smk/smk_all_artworks.json'


class Smk():
    """A model for SMK's data."""

    def __init__(self):
        """Initialize class."""
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.logger.info("Loading %s", DATAFILE)

        self.data = pd.read_json(DATAFILE).set_index('id')
        self.public_domain_images = self.data[self.data['public_domain']
                                              & self.data['has_image']]

    def get_sample(self, name: str):
        """Get a named random sample set of artworks, given a name."""
        if name == 'sample1000':
            sample = self.public_domain_images.sample(1000, random_state=SEED)

        elif name == 'bam-stratified-sample1000':
            nof_samples = 1000
            bam_objects = ['blyant', 'maleri', 'pen', 'akvarel']
            sample = self.public_domain_images.groupby(
                lambda l:
                self.public_domain_images.loc[l]['object_names'][0]['name']
                if self.public_domain_images.loc[l]['object_names'][0]['name'] in bam_objects
                else None
            ).apply(
                lambda l:
                l.sample(int(nof_samples/len(bam_objects)), random_state=SEED))

        elif name == 'sample-next5000':
            skip = 1000
            nof_samples = 5000
            sample = self.public_domain_images.sample(
                skip+nof_samples,
                random_state=SEED)[skip:]

        elif name == 'the-rest-after-next5000-not-on-iip-smk-dk':
            skip = 1000 + 5000
            nof_samples = len(self.public_domain_images)
            the_rest = self.public_domain_images.sample(
                nof_samples,
                random_state=SEED)[skip:]
            sample = the_rest[the_rest['image_native'].map(
                lambda l: urlparse(l).netloc) != "iip.smk.dk"]

        else:
            raise ValueError(f"Sample set {name} not in {self}")

        return sample

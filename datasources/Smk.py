"""Statens Museum for Kunst."""

import pandas as pd
import logging

SEED = 2019
DATAFILE = 'data/smk/smk_all_artworks.json'


class Smk():
    """A model for SMK's data."""
    def __init__(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.logger.info("Loading %s", DATAFILE)

        self.data = pd.read_json(DATAFILE).set_index('id')
        self.public_domain_images = self.data[self.data['public_domain'] & self.data['has_image']]

    def get_sample(self, name):
        """Get a named random sample set of artworks, given a name."""
        if name == 'sample1000':
            sample = self.public_domain_images.sample(1000, random_state=SEED)

        elif name == 'bam-stratified-sample1000':
            bam_objects = ['blyant', 'maleri', 'pen', 'akvarel']
            sample = self.public_domain_images.groupby(
                lambda l:
                self.public_domain_images.loc[l]['object_names'][0]['name']
                if self.public_domain_images.loc[l]['object_names'][0]['name'] in bam_objects
                else None
            ).apply(
                lambda l:
                l.sample(int(N/len(bam_objects)), random_state=SEED))

        elif name == 'sample-next5000':
            skip = 1000
            n = 5000
            sample = self.public_domain_images.sample(skip+n, random_state=SEED)[skip:]

        else:
            raise ValueError(f"Sample set {name} not in {self}")

        return sample

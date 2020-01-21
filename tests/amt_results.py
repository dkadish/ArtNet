import unittest
import pandas as pd
from datasources import Smk


class TestAmtPersonClassificationForRestOfSmkResults(unittest.TestCase):
    result_dir = 'data/amt_results/the-rest-of-smk-excluding-iip-smk-server/'
    result_files = [
        'Batch_3895760_batch_results.csv',
        'Batch_3895771_batch_results.csv',
        'Batch_3895772_batch_results.csv',
        'Batch_3895786_batch_results.csv',
        'Batch_3895787_batch_results.csv',
        'Batch_3895789_batch_results.csv',
        'Batch_3895791_batch_results.csv']

    def setUp(self):
        smk = Smk()
        self.sample = smk.get_sample('the-rest-after-next5000-not-on-iip-smk-dk')
        self.results = pd.DataFrame()
        for file in self.result_files:
            self.results = self.results.append(
                pd.read_csv(f"{self.result_dir}/{file}",
                            index_col='HITId'))
        self.results = self.results.rename(columns=lambda l: l.replace('.', '_'))

    def test_should_have_as_many_assignments_as_images_in_sample(self):
        self.assertTrue(len(self.sample) * 3 == len(self.results))

    def test_should_have_as_many_unique_hits_as_images_in_sample(self):
        self.assertTrue(len(self.sample) == self.results.index.unique().size)

    def test_results_should_have_three_assignments_per_image(self):
        self.assertTrue(
            (self.results.groupby('Input_id').size() == 3).all())

    def test_results_should_cover_all_images_in_sample_and_nothing_more(self):
        self.assertTrue(
            self.sample.index.difference(
                self.results.set_index('Input_id').index
            ).empty)

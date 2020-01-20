"""Test the SMK datasource.

Is it awkward to define many test classes in one file?

"""
import unittest
from urllib.parse import urlparse
from datasources import Smk

class TestSMK(unittest.TestCase):
    """Generic tests for the class."""
    def setUp(self):
        self.smk = Smk()

    def test_dataframe_matches_constant_shape(self):
        self.assertEqual(self.smk.data.shape, (79004, 52))

    def test_index_is_unique(self):
        self.assertTrue(self.smk.data.index.unique)


class TestSMKSampleSets(unittest.TestCase):
    """Test sampling of data."""
    def setUp(self):
        self.smk = Smk()

    # sample1000
    def test_sample1000_matches_constant_shape(self):
        sample = self.smk.get_sample('sample1000')
        self.assertEqual(sample.shape, (1000, 52))

    def test_sample1000_should_begin_and_end_with_constant_ids(self):
        sample = self.smk.get_sample('sample1000')
        self.assertEqual('1180029311_object', sample.iloc[0].name)
        self.assertEqual('1180069565_object', sample.iloc[-1].name)

    # sample-next5000
    def test_sample_next5000_should_match_constant_shape(self):
        sample = self.smk.get_sample('sample-next5000')
        self.assertEqual(sample.shape, (5000, 52))

    def test_sample_next5000_should_begin_and_end_with_constant_ids(self):
        sample = self.smk.get_sample('sample-next5000')
        self.assertEqual('1180077163_object', sample.iloc[0].name)
        self.assertEqual('1180055623_object', sample.iloc[-1].name)

    # sample-the-rest-after-next5000
    def test_sample_the_rest_after_next5000_not_on_iip_should_match_constant_shape(self):
        sample = self.smk.get_sample('the-rest-after-next5000-not-on-iip-smk-dk')
        self.assertEqual(sample.shape, (10245, 52))

    def test_sample_the_rest_after_next5000_not_on_iip_should_begin_and_end_with_constant_ids(self):
        sample = self.smk.get_sample('the-rest-after-next5000-not-on-iip-smk-dk')
        self.assertEqual('1180063742_object', sample.iloc[0].name)
        self.assertEqual('1180060575_object', sample.iloc[-1].name)

    def test_sample_the_rest_after_next5000_not_on_iip_should_not_have_objects_from_iip_smk_dk(self):
        sample = self.smk.get_sample('the-rest-after-next5000-not-on-iip-smk-dk')
        self.assertNotIn(
            True,
            sample['image_native'].map(lambda l: urlparse(l).netloc) == 'iip.smk.dk')


class TestSMKSampleSetInteractions(unittest.TestCase):
    """Test sampling of data."""
    def setUp(self):
        self.smk = Smk()

    # Samples should not overlap
    def test_sample1000_and_sample_next5000_should_not_overlap(self):
        s1000 = self.smk.get_sample('sample1000')
        s5000 = self.smk.get_sample('sample-next5000')
        self.assertTrue(s1000.join(s5000, how='inner', rsuffix='_r').empty)

    def test_sample_1000_sample_the_rest_after_next5000_not_on_iip_should_not_overlap(self):
        s1000 = self.smk.get_sample('sample1000')
        rest = self.smk.get_sample('the-rest-after-next5000-not-on-iip-smk-dk')
        self.assertTrue(s1000.join(rest, how='inner', rsuffix='_r').empty)

    def test_sample_next5000_sample_the_rest_after_next5000_not_on_iip_should_not_overlap(self):
        s5000 = self.smk.get_sample('sample-next5000')
        rest = self.smk.get_sample('the-rest-after-next5000-not-on-iip-smk-dk')
        self.assertTrue(s5000.join(rest, how='inner', rsuffix='_r').empty)


if __name__ == '__main__':
    unittest.main()

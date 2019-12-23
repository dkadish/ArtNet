"""Test the SMK datasource."""
import unittest
from datasources import Smk

class TestSMK(unittest.TestCase):
    """Generic tests for the class."""
    def setUp(self):
        self.smk = Smk()

    def test_dataframe_matches_constant_shape(self):
        self.assertEqual(self.smk.data.shape, (79004, 52))

    def test_index_is_unique(self):
        self.assertTrue(self.smk.data.index.unique)


class TestSMKSampling(unittest.TestCase):
    """Test sampling of data."""
    def setUp(self):
        self.smk = Smk()

    def test_sample1000_matches_constant_shape(self):
        sample = self.smk.get_sample('sample1000')
        self.assertEqual(sample.shape, (1000, 52))

    def test_sample1000_should_begin_and_end_with_constant_ids(self):
        sample = self.smk.get_sample('sample1000')
        self.assertEqual('1180029311_object', sample.iloc[0].name)
        self.assertEqual('1180069565_object', sample.iloc[-1].name)

    def test_sample_next5000_should_match_constant_shape(self):
        sample = self.smk.get_sample('sample-next5000')
        self.assertEqual(sample.shape, (5000, 52))

    def test_sample_next5000_should_begin_and_end_with_constant_ids(self):
        sample = self.smk.get_sample('sample-next5000')
        self.assertEqual('1180077163_object', sample.iloc[0].name)
        self.assertEqual('1180055623_object', sample.iloc[-1].name)

    def test_sample1000_and_sample_next5000_should_not_overlap(self):
        s1000 = self.smk.get_sample('sample1000')
        s5000 = self.smk.get_sample('sample-next5000')
        self.assertTrue(s1000.join(s5000, how='inner', rsuffix='_r').empty)


if __name__ == '__main__':
    unittest.main()

# Unittests for ArtNet data.

import unittest
import hashlib
import os
import pandas as pd

DATAFILE = 'data/smk/smk_all_artworks.json'


class TestSMKJsonFile(unittest.TestCase):
    """Test class against the file."""
    def test_datafile_should_be_available(self):
        self.assertTrue(os.path.exists(DATAFILE))
        self.assertTrue(os.path.isfile(DATAFILE))

    def test_md5sum_matches_constant(self):
        with(open(DATAFILE, 'rb')) as fd:
            self.assertEqual(
                hashlib.md5(fd.read()).hexdigest(),
                '83edafa71f53e51f45258c248b928e44')


class TestSMKJsonData(unittest.TestCase):
    """Test class against the data, as a Pandas dataframe."""
    def setUp(self):
        self.data = pd.read_json(DATAFILE)

    def test_dataframe_matches_constant_shape(self):
        self.assertEqual(self.data.shape, (79004, 53))

    def test_object_id_are_unique(self):
        self.assertTrue(self.data['id'].unique)

    def test_all_items_have_rights(self):
        self.assertFalse(self.data['rights'].isnull().values.any())


if __name__ == '__main__':
    unittest.main()

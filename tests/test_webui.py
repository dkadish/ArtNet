# Unittests for ArtNet web user interface.

import unittest
from webui import sample


class TestWebUi(unittest.TestCase):
    def setUp(self):
        with sample.app.test_client() as client:
            self.client = client

    def test_invalid_url_should_return_404(self):
        resp = self.client.get('/nothing_here_xxxx')
        self.assertEqual(404, resp.status_code)

    def test_hello_should_return_hello_world(self):
        resp = self.client.get('/hello_world')
        self.assertEqual(200, resp.status_code)
        self.assertEqual(b'Hello world!', resp.data)

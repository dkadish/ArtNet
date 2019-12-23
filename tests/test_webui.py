# Unittests for ArtNet web user interface.

import unittest
from io import BytesIO
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

    def test_uploading_a_file_to_form_should_return_a_redirect(self):
        """Test image upload using mock image data."""
        resp = self.client.post(
            '/upload_image',
            content_type='multipart/form-data',
            data={'image': (BytesIO(b'Image content'), 'filename.jpg')})
        self.assertEqual(302, resp.status_code)

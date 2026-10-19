import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "smoke", Path(__file__).resolve().parents[1] / "scripts/smoke_test.py"
)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class SmokeTests(unittest.TestCase):
    def test_vectors(self):
        vector = [1 / 32] * 1024
        smoke.validate_embeddings({"data": [
            {"index": 1, "embedding": vector}, {"index": 0, "embedding": vector}
        ]})

    def test_bad_vectors(self):
        for vector in ([0] * 1024, [float("nan")] * 1024, [1]):
            with self.assertRaises(ValueError):
                smoke.validate_embeddings({"data": [{"index": 0, "embedding": vector}]}, 1)

    def test_wrong_indexes(self):
        with self.assertRaises(ValueError):
            smoke.validate_embeddings({"data": [{"index": 1, "embedding": [1 / 32] * 1024}]}, 1)

    def test_media(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.wav"
            path.write_bytes(b"fixture")
            part = smoke.media_part(path, "audio")
            self.assertEqual(part["input_audio"], {"data": "Zml4dHVyZQ==", "format": "wav"})
            path = Path(directory) / "test.png"
            path.write_bytes(b"fixture")
            self.assertTrue(smoke.media_part(path, "image")["image_url"]["url"].startswith(
                "data:image/png;base64,"
            ))


if __name__ == "__main__":
    unittest.main()

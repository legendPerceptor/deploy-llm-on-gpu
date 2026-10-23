"""Exercise API contracts without loading a GPU model."""
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "qwen3_trial_http", Path(__file__).resolve().parents[1] / "scripts/qwen3_omni_trial.py")
trial = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trial)


@unittest.skipUnless(importlib.util.find_spec("fastapi"), "Run in the trial image for API tests")
class HttpTests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        captured = {}
        runner = SimpleNamespace(infer=lambda *_, **kwargs: {
            "answer": "323", "finish_reason": "stop",
            "usage": {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24}})
        with patch.dict(os.environ, {"TRIAL_API_KEY": "unit-test-only"}), \
                patch("uvicorn.run", side_effect=lambda app, **_: captured.update(app=app)):
            trial.serve(runner)
        self.client = TestClient(captured["app"])
        self.headers = {"Authorization": "Bearer unit-test-only"}
        self.body = {"model": "qwen3-omni-30b-a3b-nf4",
                     "messages": [{"role": "user", "content": "17*19"}]}

    def test_auth_required(self):
        self.assertEqual(self.client.get("/v1/models").status_code, 401)

    def test_completion(self):
        response = self.client.post("/v1/chat/completions", headers=self.headers, json=self.body)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["choices"][0]["message"]["content"], "323")

    def test_reject_unsupported_protocols(self):
        for field, value in [("stream", True),
                             ("response_format", {"type": "json_schema"})]:
            with self.subTest(field=field):
                response = self.client.post("/v1/chat/completions", headers=self.headers,
                                            json={**self.body, field: value})
                self.assertEqual(response.status_code, 400)

    def test_accepts_json_object_mode(self):
        response = self.client.post("/v1/chat/completions", headers=self.headers,
                                    json={**self.body, "response_format": {"type": "json_object"}})
        self.assertEqual(response.status_code, 200)

    def test_reject_unknown_model(self):
        response = self.client.post("/v1/chat/completions", headers=self.headers,
                                    json={**self.body, "model": "not-a-model"})
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()

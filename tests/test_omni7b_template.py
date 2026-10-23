import json
from pathlib import Path
import unittest

from jinja2 import Environment


class Omni7BTemplateTests(unittest.TestCase):
    def render(self, messages, **kwargs):
        source = (Path(__file__).resolve().parents[1] /
                  "templates/qwen2_5_omni7b_json_tools.jinja").read_text()
        return Environment().from_string(source).render(
            messages=messages, add_generation_prompt=True, **kwargs)

    def test_media_and_plain_chat(self):
        text = self.render([{"role": "user", "content": "Hello"}])
        self.assertIn("<|im_start|>user\nHello<|im_end|>", text)
        media = self.render([{"role": "user", "content": [
            {"type": "image"}, {"type": "audio"}, {"type": "video"},
        ]}])
        for token in ("<|IMAGE|>", "<|AUDIO|>", "<|VIDEO|>"):
            self.assertIn(token, media)

    def test_json_call_history_matches_parser(self):
        for arguments in ({"item": "测试"}, '{"item":"测试"}'):
            result = self.render([
                {"role": "assistant", "content": None, "tool_calls": [
                    {"function": {"name": "lookup", "arguments": arguments}}]},
                {"role": "tool", "content": '{"code":"verified"}'},
            ], tools=[{"type": "function", "function": {"name": "lookup"}}])
            call = result.split("<|im_start|>assistant\n")[1].split("<|im_end|>")[0]
            self.assertEqual(json.loads(call),
                             {"name": "lookup", "arguments": {"item": "测试"}})
            self.assertNotIn("<tool_call>", result)
            self.assertIn("Do not use Python function-call syntax", result)
            self.assertIn('<tool_response>\n{"code":"verified"}', result)

    def test_multiple_calls_are_semicolon_separated(self):
        result = self.render([{
            "role": "assistant", "content": None,
            "tool_calls": [{"function": {"name": name, "arguments": {}}}
                           for name in ("first", "second")],
        }])
        calls = result.split("<|im_start|>assistant\n")[1].split("<|im_end|>")[0]
        self.assertEqual([json.loads(c)["name"] for c in calls.split("; ")],
                         ["first", "second"])

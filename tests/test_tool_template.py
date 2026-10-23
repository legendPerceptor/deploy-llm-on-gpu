import json
from pathlib import Path
import unittest

from jinja2 import Environment


class ToolTemplateTests(unittest.TestCase):
    def render(self, messages, **kwargs):
        source = (Path(__file__).resolve().parents[1] /
                  "templates/qwen2_5_omni_tools.jinja").read_text()
        return Environment().from_string(source).render(
            messages=messages, add_generation_prompt=True, **kwargs)

    def test_plain_chat(self):
        self.assertEqual(self.render([{"role": "user", "content": "Hello"}]),
                         "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                         "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n")

    def test_media(self):
        result = self.render([{"role": "system", "content": "Keep this"},
                              {"role": "user", "content": [
                                  {"type": "text", "text": "Describe:"},
                                  {"type": "image"}, {"type": "audio"}, {"type": "video"},
                              ]}])
        self.assertIn("system\nKeep this<|im_end|>", result)
        for token in ("<|vision_bos|><|IMAGE|><|vision_eos|>",
                      "<|audio_bos|><|AUDIO|><|audio_eos|>",
                      "<|vision_bos|><|VIDEO|><|vision_eos|>"):
            self.assertIn(token, result)

    def test_tool_history(self):
        for arguments in ({"item": "测试"}, '{"item": "测试"}'):
            with self.subTest(arguments=arguments):
                result = self.render([
                    {"role": "user", "content": "Look it up"},
                    {"role": "assistant", "content": None, "tool_calls": [
                        {"function": {"name": "lookup", "arguments": arguments}}]},
                    {"role": "tool", "content": "result-one"},
                    {"role": "tool", "content": "result-two"},
                ], tools=[{"type": "function", "function": {"name": "lookup"}}])
                self.assertIn("<tools>", result)
                call = result.split("<tool_call>\n")[-1].split("\n</tool_call>")[0]
                self.assertEqual(json.loads(call),
                                 {"name": "lookup", "arguments": {"item": "测试"}})
                self.assertIn("<tool_response>\nresult-one\n</tool_response>", result)
                self.assertEqual(result.count("<|im_start|>user"), 2)

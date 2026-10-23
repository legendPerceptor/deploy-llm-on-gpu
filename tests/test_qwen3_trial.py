import importlib.util
import json
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "qwen3_omni_trial.py"
spec = importlib.util.spec_from_file_location("qwen3_trial", SCRIPT)
trial = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trial)


class NormalizeTests(unittest.TestCase):
    def test_text(self):
        self.assertEqual(trial.normalize_messages([{"role": "user", "content": "hello"}]),
                         [{"role": "user", "content": "hello"}])

    def test_image_url(self):
        result = trial.normalize_messages([{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]}])
        self.assertEqual(result[0]["content"][0]["image"], "data:image/png;base64,abc")

    def test_reject_unknown_role(self):
        with self.assertRaises(ValueError):
            trial.normalize_messages([{"role": "unknown", "content": "result"}])

    def test_tool_history_preserved(self):
        messages = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_a", "type": "function",
                 "function": {"name": "Search", "arguments": '{"query":"x"}'}}]},
            {"role": "tool", "tool_call_id": "call_a", "content": "evidence"},
        ]
        normalized = trial.normalize_messages(messages)
        self.assertEqual(normalized[0]["tool_calls"], messages[0]["tool_calls"])
        self.assertEqual(normalized[0]["content"], "")
        self.assertEqual(normalized[1], messages[1])

    def test_reject_unknown_content(self):
        with self.assertRaises(ValueError):
            trial.normalize_messages([{"role": "user", "content": [{"type": "video"}]}])

    def test_does_not_modify_input(self):
        original = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
        result = trial.normalize_messages(original)
        result[0]["content"][0]["text"] = "modified"
        self.assertEqual(original[0]["content"][0]["text"], "hello")


class ProtocolTests(unittest.TestCase):
    tools = [{"type": "function", "function": {
        "name": "Search", "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}}, "required": ["query"], "additionalProperties": False}}}]

    def test_real_tool_markup(self):
        answer, calls = trial.parse_generated_output(
            '<tool_call>{"name":"Search","arguments":{"query":"clay"}}</tool_call>',
            tools=self.tools)
        self.assertEqual(answer, "")
        self.assertEqual(calls[0]["function"]["name"], "Search")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"query": "clay"})
        self.assertTrue(calls[0]["id"].startswith("call_"))

    def test_no_call_fabricated_from_plain_json(self):
        text = '{"name":"Search","arguments":{"query":"clay"}}'
        answer, calls = trial.parse_generated_output(text, tools=self.tools)
        self.assertEqual(answer, text)
        self.assertEqual(calls, [])

    def test_invalid_calls_fail(self):
        for text in [
            '<tool_call>{"name":"Search","arguments":{}}</tool_call>',
            '<tool_call>{"name":"Other","arguments":{"query":"x"}}</tool_call>',
            '<tool_call>{"name":"Search","arguments":{"query":2}}</tool_call>',
            '<tool_call>{"name":"Search"',
        ]:
            with self.subTest(text=text), self.assertRaises(trial.ModelProtocolError):
                trial.parse_generated_output(text, tools=self.tools)

    def test_required_call_is_not_faked(self):
        with self.assertRaises(trial.ModelProtocolError):
            trial.parse_generated_output("I cannot call it", tools=self.tools, tool_choice="required")

    def test_none_rejects_calls(self):
        with self.assertRaises(trial.ModelProtocolError):
            trial.parse_generated_output(
                '<tool_call>{"name":"Search","arguments":{"query":"x"}}</tool_call>',
                tools=self.tools, tool_choice="none")

    def test_json_mode_validates_object(self):
        answer, calls = trial.parse_generated_output(
            '```json\n{"description": "An apple"}\n```', response_format={"type": "json_object"})
        self.assertEqual(json.loads(answer), {"description": "An apple"})
        self.assertEqual(calls, [])
        for text in ["not json", "[]", '{"description":']:
            with self.subTest(text=text), self.assertRaises(trial.ModelProtocolError):
                trial.parse_generated_output(text, response_format={"type": "json_object"})


@unittest.skipUnless(importlib.util.find_spec("torch"), "Requires torch for numerical hook checks")
class LogitsTests(unittest.TestCase):
    def test_last_token_matches_full_head(self):
        import torch
        torch.manual_seed(0)
        head = torch.nn.Linear(32, 64).eval()
        hidden = torch.randn(1, 256, 32)
        expected = head(hidden)[:, -1:, :]
        head.register_forward_pre_hook(trial.last_token_logits_input)
        actual = head(hidden)
        self.assertEqual(tuple(actual.shape), (1, 1, 64))
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_training_is_not_sliced(self):
        import torch
        head = torch.nn.Linear(4, 8).train()
        head.register_forward_pre_hook(trial.last_token_logits_input)
        self.assertEqual(tuple(head(torch.randn(1, 5, 4)).shape), (1, 5, 8))

    def test_decode_step_shape_unchanged(self):
        import torch
        head = torch.nn.Linear(4, 8).eval()
        head.register_forward_pre_hook(trial.last_token_logits_input)
        self.assertEqual(tuple(head(torch.randn(1, 1, 4)).shape), (1, 1, 8))

if __name__ == "__main__":
    unittest.main()

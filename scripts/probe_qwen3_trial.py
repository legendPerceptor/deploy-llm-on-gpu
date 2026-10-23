"""Probe the trial HTTP boundary, including data-URI images and raw audio."""
import argparse
import base64
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request


def request(base, key, path, payload=None, timeout=1200):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8003")
    parser.add_argument("--result-dir", type=Path, default=Path("results/qwen3-omni-20261022"))
    args = parser.parse_args()
    key = os.environ["API_KEY"]
    media = args.result_dir / "media"
    image = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," +
             base64.b64encode((media / "cars.jpg").read_bytes()).decode()}}
    audio = {"type": "input_audio", "input_audio": {"format": "wav", "data":
             base64.b64encode((media / "living_room.wav").read_bytes()).decode()}}
    filler = [f"Record {i:04d}: This is ordinary background information about household items."
              for i in range(1100)]
    filler.insert(550, "Special fact: Alice placed the blue notebook in the kitchen drawer. "
                  "The retrieval code is M3-7429.")
    long_prompt = "\n".join(filler) + (
        '\nReturn only JSON with keys "code" and "location" for the special fact. '
        'Ignore the ordinary records. Do not add Markdown fences.')
    cases = {
        "text": ([{"type": "text", "text": "计算17乘以19，只回答数字。"}], 32),
        "image_data_uri": ([image, {"type": "text", "text": "How many cars are shown? "
                                  "Name their colors briefly."}], 128),
        "audio_base64": ([audio, {"type": "text", "text": "Transcribe the audible English speech."}], 128),
        "image_audio_base64": ([image, audio, {"type": "text", "text":
            'Return JSON with keys "visual" and "audio" describing the image and the speech '
            'separately. These inputs are unrelated. No Markdown fences.'}], 256),
        "long_context": ([{"type": "text", "text": long_prompt}], 8192),
    }
    output = {"protocol": {}, "cases": {}}
    for name, auth, path, body, expected in [
        ("auth_required", "", "/v1/models", None, 401),
        ("models", key, "/v1/models", None, 200),
        ("reject_named_tool_choice", key, "/v1/chat/completions",
         {"model": "qwen3-omni-30b-a3b-nf4", "messages": [],
          "tool_choice": {"type": "function", "function": {"name": "Search"}}}, 400),
        ("reject_streaming", key, "/v1/chat/completions",
         {"model": "qwen3-omni-30b-a3b-nf4", "messages": [], "stream": True}, 400),
    ]:
        status, result = request(args.base, auth, path, body)
        output["protocol"][name] = {"status": status, "expected": expected, "result": result}
    for name, (content, budget) in cases.items():
        started = time.time()
        status, response = request(args.base, key, "/v1/chat/completions", {
            "model": "qwen3-omni-30b-a3b-nf4", "messages": [{"role": "user", "content": content}],
            "max_tokens": budget,
        })
        record = {"status": status, "seconds": time.time() - started, "max_tokens": budget,
                  "response": response}
        if status == 200:
            choice = response["choices"][0]
            record["nonempty"] = bool(choice["message"]["content"].strip())
            record["normal_stop"] = choice["finish_reason"] == "stop"
            if name == "long_context":
                try:
                    value = json.loads(choice["message"]["content"])
                    record["needle_correct"] = value == {"code": "M3-7429", "location": "kitchen drawer"}
                except ValueError:
                    record["needle_correct"] = False
        output["cases"][name] = record
        print(json.dumps({name: record}, ensure_ascii=False), flush=True)
        (args.result_dir / "http-probes.json").write_text(json.dumps(output, ensure_ascii=False, indent=2))
    assert all(p["status"] == p["expected"] for p in output["protocol"].values())
    assert all(p.get("nonempty") and p.get("normal_stop") for p in output["cases"].values())
    assert output["cases"]["long_context"].get("needle_correct")


if __name__ == "__main__":
    main()

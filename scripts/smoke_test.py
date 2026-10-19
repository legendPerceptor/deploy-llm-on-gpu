#!/usr/bin/env python3
"""Standard-library-only endpoint checks; not a benchmark or quality evaluation."""
import argparse
import base64
import json
import math
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request


def request(base, path, key, payload=None, timeout=300):
    headers = {"Authorization": f"Bearer {key}"}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode()
    req = urllib.request.Request(base.rstrip("/") + path, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def media_part(path, kind):
    path = Path(path)
    if path.stat().st_size > 10 * 1024 * 1024:
        raise ValueError("Smoke-test media must be <=10 MiB; use a small image/short clip.")
    data = base64.b64encode(path.read_bytes()).decode()
    suffix = path.suffix.lower()
    if kind == "audio":
        if suffix not in {".wav", ".mp3"}:
            raise ValueError("Audio smoke test accepts .wav or .mp3.")
        return {"type": "input_audio", "input_audio": {"data": data, "format": suffix[1:]}}
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(suffix)
    if not mime:
        raise ValueError("Image smoke test accepts PNG or JPEG.")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


def validate_embeddings(response, count=2):
    rows = sorted(response["data"], key=lambda row: row["index"])
    if len(rows) != count or [row["index"] for row in rows] != list(range(count)):
        raise ValueError("Unexpected embedding count or indexes")
    for row in rows:
        vector = row["embedding"]
        if len(vector) != 1024 or not all(math.isfinite(x) for x in vector):
            raise ValueError("Expected 1024 finite BGE-M3 embedding values")
        norm = math.sqrt(sum(x * x for x in vector))
        if not 0.98 <= norm <= 1.02:
            raise ValueError(f"Expected normalized embedding; norm={norm}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--omni-url", default="http://127.0.0.1:8000")
    parser.add_argument("--embedding-url", default="http://127.0.0.1:8001")
    parser.add_argument("--model", default="omni-3b")
    parser.add_argument("--image")
    parser.add_argument("--audio")
    args = parser.parse_args()
    key = os.environ.get("API_KEY")
    if not key:
        parser.error("Export API_KEY matching .env before running.")
    models = request(args.omni_url, "/v1/models", key)
    if args.model not in {item["id"] for item in models["data"]}:
        raise ValueError(f"Model {args.model!r} not advertised by endpoint")
    messages = [{"role": "user", "content": "Reply with one short greeting."}]
    tests = [("text", messages)]
    parts = []
    for kind in ("image", "audio"):
        path = getattr(args, kind)
        if path:
            part = media_part(path, kind)
            parts.append(part)
            tests.append((kind, [{"role": "user", "content": [
                {"type": "text", "text": f"Briefly describe this {kind}."}, part
            ]}]))
    if len(parts) == 2:
        tests.append(("image+audio", [{"role": "user", "content": [
            {"type": "text", "text": "Briefly describe the image and audio."}, *parts
        ]}]))
    for label, messages in tests:
        result = request(args.omni_url, "/v1/chat/completions", key, {
            "model": args.model, "messages": messages,
            "max_tokens": 128, "temperature": 0, "stream": False,
        })
        text = result["choices"][0]["message"]["content"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Empty/non-text response for {label}")
        print(f"PASS {label}: {text}")
    result = request(args.embedding_url, "/v1/embeddings", key, {
        "model": "BAAI/bge-m3", "input": ["A small GPU server.", "一台小型 GPU 服务器。"],
        "encoding_format": "float",
    })
    validate_embeddings(result)
    print("PASS embeddings: two finite, normalized 1024-dimensional vectors")
    print("Not tested: video, speech generation, throughput, benchmark correctness.")


if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.read().decode(errors='replace')}", file=sys.stderr)
        sys.exit(1)
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)

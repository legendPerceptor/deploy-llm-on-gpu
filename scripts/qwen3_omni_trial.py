"""Single-request, localhost trial server; not a production vLLM replacement.

Official Qwen3-Omni weights, NF4/FP16, text output only. The audio/image
encoders stay FP16. Uses the official tool template with validated output
parsing. JSON mode is post-validated, not grammar-constrained decoding.
"""
import argparse
import base64
import io
import json
import os
from pathlib import Path
import re
import threading
import time
import traceback
import uuid

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
REVISION = "26291f793822fb6be9555850f06dfe95f2d7e695"


class ModelProtocolError(RuntimeError):
    """Generated output violated the requested protocol; never invent a call."""


def last_token_logits_input(module, inputs):
    """Generation uses only the last position; avoid a [batch, context, vocab] tensor.

    Installed only on this inference-only Thinker. Speech generation and
    teacher-forced scoring are not supported by this trial server.
    """
    hidden = inputs[0]
    if not module.training and hidden.ndim == 3 and hidden.shape[1] > 1:
        return (hidden[:, -1:, :], *inputs[1:])
    return inputs


def parse_generated_output(text, *, tools=None, tool_choice="auto", response_format=None):
    calls = []
    if "<tool_call>" in text or "</tool_call>" in text:
        matches = list(re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S))
        remainder = re.sub(r"<tool_call>\s*.*?\s*</tool_call>", "", text, flags=re.S).strip()
        if not matches or "<tool_call>" in remainder or "</tool_call>" in remainder:
            raise ModelProtocolError("Incomplete tool-call markup")
        allowed = {t["function"]["name"]: t["function"] for t in (tools or [])}
        if tool_choice == "none":
            raise ModelProtocolError("Model called a tool when tool_choice=none")
        from jsonschema import validate
        for match in matches:
            try:
                item = json.loads(match.group(1))
                name, arguments = item["name"], item["arguments"]
                if name not in allowed or not isinstance(arguments, dict):
                    raise ValueError("Unknown tool or non-object arguments")
                validate(arguments, allowed[name].get("parameters", {"type": "object"}))
            except Exception as error:
                raise ModelProtocolError(f"Invalid generated tool call: {error}") from error
            calls.append({"id": "call_" + uuid.uuid4().hex, "type": "function",
                          "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}})
        text = remainder
    if tool_choice == "required" and not calls:
        raise ModelProtocolError("Model did not produce the required tool call")
    if response_format:
        candidate = text.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", candidate, re.S)
        if fenced:
            candidate = fenced.group(1).strip()
        try:
            obj = json.loads(candidate)
            if not isinstance(obj, dict):
                raise ValueError("Expected a JSON object")
        except ValueError as error:
            raise ModelProtocolError(f"Model did not return a JSON object: {error}") from error
        text = json.dumps(obj, ensure_ascii=False)
    return text, calls


def normalize_messages(messages):
    """Accept local trial content or OpenAI image/audio content."""
    normalized = []
    for message in messages:
        if message["role"] not in ("system", "user", "assistant", "tool"):
            raise ValueError("Unsupported message role")
        content = message.get("content") or ""
        # Keep text-only history as strings for the official template's
        # last-user-turn detection and tool-response rendering.
        if isinstance(content, str):
            item = {"role": message["role"], "content": content}
            for key in ("tool_calls", "tool_call_id"):
                if key in message:
                    item[key] = message[key]
            normalized.append(item)
            continue
        parts = []
        for part in content:
            kind = part["type"]
            if kind == "image_url":
                url = part["image_url"]
                parts.append({"type": "image", "image": url["url"] if isinstance(url, dict) else url,
                              "max_pixels": 200704})
            elif kind == "input_audio":
                import soundfile as sf
                import librosa
                audio, sr = sf.read(io.BytesIO(base64.b64decode(part["input_audio"]["data"])))
                if audio.ndim == 2:
                    audio = audio.mean(axis=1)
                parts.append({"type": "audio", "audio": librosa.resample(
                    audio, orig_sr=sr, target_sr=16000).astype("float32")})
            elif kind in ("text", "image", "audio"):
                parts.append(dict(part))
            else:
                raise ValueError(f"Unsupported content type: {kind}")
        item = {"role": message["role"], "content": parts}
        if message.get("tool_calls"):
            item["tool_calls"] = message["tool_calls"]
        normalized.append(item)
    return normalized


class Runner:
    def __init__(self, result_dir, gpu_gib):
        import torch
        import bitsandbytes as bnb
        from transformers.integrations import sdpa_attention
        from transformers import (
            AutoConfig, BitsAndBytesConfig, Qwen3OmniMoeForConditionalGeneration,
            Qwen3OmniMoeProcessor,
        )
        self.torch = torch
        # Pinned transformers 4.57.6 selects native GQA based on torch version.
        # T4 cannot run FlashAttention, and torch's efficient SDPA cannot accept
        # unequal Q/KV head counts. Native GQA thus falls back to quadratic math
        # attention (observed 37 GiB allocation for ~18K input). Explicit repeat_kv
        # preserves GQA semantics and makes efficient SDPA available on Turing.
        # This override is process-local; no installed dependency is modified.
        sdpa_attention.use_gqa_in_sdpa = lambda attention_mask, key: False
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        config = AutoConfig.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
        # Avoid even constructing/loading the speech generator.
        config.enable_audio_output = False
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
            llm_int8_skip_modules=["lm_head", "gate", "visual", "audio_tower"],
        )
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            MODEL, revision=REVISION, local_files_only=True, config=config,
            dtype=torch.float16, device_map="balanced",
            max_memory={i: f"{gpu_gib}GiB" for i in range(torch.cuda.device_count())},
            quantization_config=quant, attn_implementation="sdpa",
        ).eval()
        # transformers 4.57.6's Omni Thinker lacks logits_to_keep support.
        # Its full-context lm_head attempted an 8.61-GiB allocation in real
        # multi-turn M3 QA. Autoregressive generation consumes only logits[:, -1].
        self.model.thinker.lm_head.register_forward_pre_hook(last_token_logits_input)
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(
            MODEL, revision=REVISION, local_files_only=True)
        self.eos_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        # The wrapper's EOS differs from thinker.generation_config; pass and
        # record the same value instead of misclassifying normal stops as length.
        if self.eos_token_id is None:
            raise RuntimeError("Missing chat end token")
        torch.cuda.empty_cache()
        devices = {str(v) for v in self.model.hf_device_map.values()}
        if devices & {"cpu", "disk"}:
            raise RuntimeError(f"Trial requires GPU-only weights, got {devices}")
        metadata = {
            "model": MODEL, "revision": REVISION, "quantization": "bitsandbytes NF4 double quant",
            "compute_dtype": "float16", "attention": "efficient_sdpa_explicit_repeat_kv",
            "logits_mode": "last_token_only",
            "talker": self.model.has_talker,
            "load_seconds": time.time() - started,
            "device_map": self.model.hf_device_map,
            "linear4bit_count": sum(isinstance(m, bnb.nn.Linear4bit) for m in self.model.modules()),
            "memory": self.memory(),
        }
        (self.result_dir / "loaded.json").write_text(json.dumps(metadata, indent=2))
        print(json.dumps({"loaded": metadata}), flush=True)

    def memory(self):
        return [{"gpu": i, "allocated_gib": self.torch.cuda.memory_allocated(i) / 2**30,
                 "reserved_gib": self.torch.cuda.memory_reserved(i) / 2**30,
                 "peak_gib": self.torch.cuda.max_memory_allocated(i) / 2**30}
                for i in range(self.torch.cuda.device_count())]

    def infer(self, messages, max_tokens=256, *, tools=None, tool_choice="auto",
              response_format=None, temperature=0.0):
        from qwen_omni_utils import process_mm_info
        from torch.nn.attention import SDPBackend, sdpa_kernel
        if not 1 <= max_tokens <= 8192:
            raise ValueError("max_tokens must be between 1 and 8192")
        for i in range(self.torch.cuda.device_count()):
            self.torch.cuda.reset_peak_memory_stats(i)
        started = time.time()
        messages = normalize_messages(messages)
        instruction = ""
        if response_format:
            instruction = "Return exactly one JSON object. No Markdown fences or explanatory text."
        elif tool_choice == "required":
            instruction = "You must respond by calling one of the supplied tools."
        if instruction:
            if messages and messages[0]["role"] == "system":
                old = messages[0]["content"]
                messages[0]["content"] = (old + "\n" + instruction if isinstance(old, str)
                                          else [*old, {"type": "text", "text": instruction}])
            else:
                messages.insert(0, {"role": "system", "content": instruction})
        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            tools=tools if tool_choice != "none" else None)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
        inputs = self.processor(text=text, audio=audios, images=images, videos=videos,
                                return_tensors="pt", padding=True, use_audio_in_video=False)
        inputs = inputs.to(self.model.device).to(self.torch.float16)
        prompt_tokens = inputs["input_ids"].shape[1]
        if prompt_tokens + max_tokens > 32768:
            raise ValueError("Trial context cap is 32768 tokens including requested output")
        # Fail explicitly if a shape is unsupported instead of silently falling
        # back to an O(sequence_length**2) allocation that cannot fit on a T4.
        sampling = {"thinker_do_sample": temperature > 0}
        if temperature > 0:
            sampling["thinker_temperature"] = temperature
        with self.torch.inference_mode(), sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            result = self.model.generate(
                **inputs, return_audio=False, use_audio_in_video=False,
                thinker_max_new_tokens=max_tokens, **sampling,
                thinker_eos_token_id=self.eos_token_id,
                thinker_return_dict_in_generate=True,
            )
        if isinstance(result, tuple):
            result = result[0]
        ids = result.sequences if hasattr(result, "sequences") else result
        generated = ids[:, prompt_tokens:]
        raw_answer = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        last_token_id = generated[0, -1].item()
        reason = "stop" if last_token_id == self.eos_token_id else "length"
        if reason == "length" and (tools or response_format):
            raise ModelProtocolError("Structured output exhausted max_tokens")
        answer, calls = parse_generated_output(
            raw_answer, tools=tools, tool_choice=tool_choice, response_format=response_format)
        if calls:
            reason = "tool_calls"
        report = {
            "answer": answer, "finish_reason": reason, "seconds": time.time() - started,
            "raw_answer": raw_answer, "tool_calls": calls, "temperature": temperature,
            "last_token_id": last_token_id, "eos_token_id": self.eos_token_id,
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": generated.shape[1],
                      "total_tokens": prompt_tokens + generated.shape[1]},
            "memory": self.memory(),
        }
        print(json.dumps(report, ensure_ascii=False), flush=True)
        return report

    def smoke(self):
        media = self.result_dir / "media"
        cases = {
            "text": [{"type": "text", "text": "计算17乘以19。只回答数字。"}],
            "image": [{"type": "image", "image": str(media / "cars.jpg"), "max_pixels": 200704},
                      {"type": "text", "text": "Describe what is visibly in this image in one sentence."}],
            "audio": [{"type": "audio", "audio": str(media / "living_room.wav")},
                      {"type": "text", "text": "Transcribe the audible English speech. "
                       "Do not add speaker names or invent inaudible words."}],
            "image_audio": [
                {"type": "image", "image": str(media / "living_room.jpg"), "max_pixels": 200704},
                {"type": "audio", "audio": str(media / "living_room.wav")},
                {"type": "text", "text": "Summarize the visible scene and audible speech separately. "
                 "Output JSON with keys visual and audio; do not invent information."}],
        }
        reports = {}
        for name, content in cases.items():
            print(f"SMOKE_START {name}", flush=True)
            try:
                reports[name] = self.infer([{"role": "user", "content": content}])
                reports[name]["nonempty"] = bool(reports[name]["answer"].strip())
            except Exception:
                reports[name] = {"error": traceback.format_exc()}
                print(reports[name]["error"], flush=True)
            (self.result_dir / "smoke.json").write_text(
                json.dumps(reports, indent=2, ensure_ascii=False))
        return reports


def serve(runner):
    from fastapi import FastAPI, HTTPException, Request
    import uvicorn
    app = FastAPI()
    lock = threading.Lock()
    token = os.environ.get("TRIAL_API_KEY")
    if not token:
        raise RuntimeError("TRIAL_API_KEY is required")

    def authorize(request):
        import hmac
        if not hmac.compare_digest(request.headers.get("authorization", ""), f"Bearer {token}"):
            raise HTTPException(401, "Invalid API key")

    @app.get("/health")
    def health():
        return {"status": "ok", "trial": True}

    @app.get("/v1/models")
    def models(request: Request):
        authorize(request)
        return {"object": "list", "data": [{"id": "qwen3-omni-30b-a3b-nf4",
                                          "object": "model", "owned_by": "local"}]}

    @app.post("/v1/chat/completions")
    def completion(body: dict, request: Request):
        authorize(request)
        if body.get("stream"):
            raise HTTPException(400, "Trial does not implement streaming")
        choice = body.get("tool_choice", "auto")
        if choice not in ("auto", "none", "required"):
            raise HTTPException(400, "Named tool choice is not supported")
        response_format = body.get("response_format")
        if response_format and response_format != {"type": "json_object"}:
            raise HTTPException(400, "Only response_format=json_object is supported")
        if response_format and body.get("tools"):
            raise HTTPException(400, "Tools and JSON response_format cannot be combined in this trial")
        if body.get("model") != "qwen3-omni-30b-a3b-nf4":
            raise HTTPException(400, "Unknown model")
        if not lock.acquire(blocking=False):
            raise HTTPException(429, "Trial permits one request at a time")
        try:
            temperature = float(body.get("temperature", 0.0))
            if not 0 <= temperature <= 2:
                raise ValueError("temperature must be between 0 and 2")
            result = runner.infer(
                body["messages"], int(body.get("max_tokens", 256)),
                tools=body.get("tools"), tool_choice=choice,
                response_format=response_format, temperature=temperature)
            message = {"role": "assistant", "content": result["answer"] or None}
            if result.get("tool_calls"):
                message["tool_calls"] = result["tool_calls"]
            return {"id": f"trial-{time.time_ns()}", "object": "chat.completion",
                    "created": int(time.time()), "model": body["model"],
                    "choices": [{"index": 0, "message": message,
                                 "finish_reason": result["finish_reason"]}],
                    "usage": result["usage"]}
        except ModelProtocolError as error:
            raise HTTPException(502, str(error)) from error
        except (ValueError, KeyError) as error:
            raise HTTPException(400, str(error)) from error
        finally:
            lock.release()

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default="/work/results/qwen3-omni-20261022")
    parser.add_argument("--gpu-gib", type=int, default=12)
    parser.add_argument("--serve", action="store_true")
    args = parser.parse_args()
    runner = Runner(args.result_dir, args.gpu_gib)
    reports = runner.smoke()
    if not all(r.get("nonempty") and r.get("finish_reason") == "stop" for r in reports.values()):
        raise RuntimeError("Startup smoke failed; inspect smoke.json before serving")
    if args.serve:
        serve(runner)

# Qwen3-Omni on four Tesla T4 GPUs — 2026-10-22

## Result

**The official `Qwen/Qwen3-Omni-30B-A3B-Instruct` checkpoint runs on this
machine with on-load NF4 quantization and speech generation disabled.**
This is a Transformers feasibility server, not a production vLLM deployment.

- Hardware: four T4 GPUs, 15360 MiB visible memory each.
- Checkpoint revision: `26291f793822fb6be9555850f06dfe95f2d7e695`.
- All 15 safetensors shards (70,523,299,202 bytes) passed SHA-256 checks
  against the official repository's LFS metadata.
- Transformers 4.57.6, bitsandbytes 0.48.2, qwen-omni-utils 0.0.8, av 19.0.2;
  PyTorch 2.8.0+cu128 inherited from `ogmemory-omni:local`.
- NF4 double quantization, FP16 compute. Vision/audio encoders, output head,
  and routing gates remain unquantized. 18,624 Linear4bit modules loaded.
- `enable_audio_output=False`: image/audio **inputs remain enabled**;
  Talker and code2wav are not loaded. All unused checkpoint keys were verified
  to belong to Talker/code2wav, not the Thinker.
- Layers distributed across all four GPUs using Accelerate `device_map="balanced"`.
  **This is not tensor parallelism**, and no model weights were offloaded to CPU/disk.
- One request at a time. Approximately 17.8 GiB of active model tensors.
  With the final allocator settings, observed short-request GPU usage was about
  19 GiB total; after the 20K-input test, reserved/process usage was about 30 GiB.
  Do not confuse active tensor size with process peak memory requirements.

### Actual inference checks

| Check | Result |
|---|---|
| Text arithmetic | `17*19` → `323` |
| Real image | Identified four cars and their colors |
| Real 10-second audio | Produced speech transcription; contains word errors |
| Image + audio | Both direct pipeline input and OpenAI-style base64 HTTP input completed |
| 17,661-token input | Correct needle values, 29 output tokens, normal EOS, about 40 seconds |
| 20,874-token input | Correct **bare JSON**, 21 output tokens, normal EOS, about 40 seconds |
| Output budget | Both long-input requests accepted `max_tokens=8192`; neither needed that many output tokens |
| HTTP boundary | Auth, model listing, unsupported tools and streaming checks behaved as specified |

The 20,874-token request used an explicit system instruction requiring a JSON
object without Markdown fences. Its response was:

```json
{"code": "M3-7429", "location": "kitchen drawer"}
```

**A failed strict-JSON test is deliberately retained.** The earlier 17,661-token
request retrieved the correct facts but wrapped JSON in Markdown despite the
user prompt forbidding fences. `http-probes.log` therefore ends with an assertion
failure. This is a formatting failure, not a context overflow or failed
retrieval. Do not report that probe suite as completely passing.

The synthetic long-input test is not a LoCoMo accuracy benchmark. Audio
transcriptions contain errors, fine-grained vehicle identification is not
consistently correct, and some generated scene/audio details are unreliable.
No claim of better memory quality than the existing models is justified yet.

### Important fixes discovered during the experiment

1. **Default SDPA was unsuitable for long input on T4.** The 17,661-token request
   initially attempted a 37.18 GiB allocation and failed. Transformers selected
   native GQA, but the efficient PyTorch backend could not accept unequal Q/KV
   head counts, and FlashAttention was unavailable for this path on T4.
   The isolated server now makes Transformers explicitly repeat KV heads and
   forces `SDPBackend.EFFICIENT_ATTENTION`, disallowing the quadratic math fallback.
   A numerical check against native-GQA math attention passed with maximum
   absolute FP16 error 0.001953125 (`atol=rtol=0.003`).
   The override is process-local, not a modification to installed package files.
2. **Allocator fragmentation:** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
   reduced short-request process usage from roughly 44 GiB to roughly 19 GiB.
3. **EOS accounting:** the model wrapper defaults to chat EOS 151645, differing
   from the nested Thinker's generation configuration. The server now explicitly
   supplies and records the same EOS; early logs in `attempt1/` incorrectly label
   normal short answers as `length`. Final runs verify actual EOS token IDs.
4. **Build networking:** online Docker builds could not reach PyPI reliably.
   The final image builds offline using a named wheel-directory build context.
5. **Full-context logits in later M3 QA:** the pinned Thinker implementation
   computes the vocabulary projection for every prompt position. A real M3
   request attempted an additional 8.61 GiB allocation on the last GPU.
   The inference-only server now slices the input to `lm_head` to its last
   position, which is the only logits position autoregressive generation uses.
   Numerical regression tests and a real **30,407-input-token** request passed.
   This optimization is not intended for training, teacher-forced scoring,
   or speech generation. A subsequent repeated-search context overflow was
   fixed in oGMemory by referencing already-present identical evidence, not by
   dropping new evidence or claiming a larger context window.

## Files and evidence

Implementation:

- `Dockerfile.qwen3-trial`
- `compose.qwen3-trial.yaml`
- `scripts/qwen3_omni_trial.py`
- `scripts/probe_qwen3_trial.py`
- `scripts/verify_qwen3_snapshot.py`
- `tests/test_qwen3_trial.py`
- `tests/test_qwen3_trial_http.py`

Ignored runtime artifacts under `results/qwen3-omni-20261022/`:

- `sha256.json`, `remote-files.json`: checkpoint verification.
- `loaded.json`, `smoke.json`, `runtime-final.log`: final configuration and direct multimodal checks.
- `http-probes.json`, `http-probes.log`: HTTP checks, including the strict-JSON failure.
- `long-context-20k.json`: successful longer request with an explicit system prompt.
- `sdpa-probe.log`, `sdpa-equivalence.log`: backend compatibility and numerical comparison.
- `attempt1/`, `attempt2/`: earlier accounting error and genuine long-input OOM.
- `gpu-probes.csv`, `gpu-after-long.csv`: observed GPU usage.
- `final-unit-tests.log`: container run, 24 passed / 1 skipped (no Docker CLI).
- `host-unit-tests.log`: host run, 21 passed / 4 skipped (no FastAPI).
  Together these execute all 25 tests, including Compose and HTTP contract checks.
- `restored-*-smoke.log`: post-experiment checks of the original services.

## Follow-up: M3 protocol adapter

The later full-video M3 experiment extends the initial feasibility server:

- Uses the checkpoint's **official tool chat template**, preserving assistant
  tool-call and tool-response history.
- Parses actual `<tool_call>` output, validates the function name and arguments
  against the supplied schema, and returns OpenAI-style tool calls with IDs.
  It never invents a call from a plain answer. `auto`, `none`, and `required`
  are supported; named forced calls are not.
- A real `lookup_test_code` round trip passed: the model called the tool and then
  used `TOOL_RESULT_7391` supplied in the tool result.
- `response_format={"type":"json_object"}` adds an instruction and validates the
  completed model output. A valid Markdown fence can be removed; invalid JSON
  fails with HTTP 502. This is **post-validation, not constrained decoding**.
- Model protocol failures remain failures; no fabricated Search/Answer results
  or incomplete structured generations are accepted.
- An explicit `enable_thinking=False` template experiment yielded an empty
  audio-only smoke answer and was blocked by startup checks. The implementation
  reverted to the official template default and passed the audio smoke again.

Evidence for this extension is separate from the original trial:
`results/m3-sample1-20261022/qwen3-runtime/`,
`qwen3-tool-smoke.json`, and `qwen3-protocol-tests.log`.
The M3 comparison uses existing Omni-7B memories for QA; this is **not** a
full-video memory ingestion benchmark performed by 30B.

## Reproduce / start on demand

Prerequisites: existing `ogmemory-omni:local` base image, existing
`ogmemory-gpu_hf-cache` volume, and the deployment `.env` with `API_KEY`.
Run commands from this directory. The checkpoint, wheels, and media already
exist on the machine after this experiment.

If recreating the wheels:

```bash
mkdir -p results/qwen3-omni-20261022/wheels
docker run --rm --entrypoint python3 \
  -v "$PWD/results/qwen3-omni-20261022/wheels:/wheels" \
  ogmemory-omni:local -m pip download --no-deps -d /wheels \
  transformers==4.57.6 bitsandbytes==0.48.2 qwen-omni-utils==0.0.8 av==19.0.2
docker compose -f compose.qwen3-trial.yaml build
```

If recreating the checkpoint cache:

```bash
docker run --rm --entrypoint python3 -e HF_HUB_DISABLE_XET=1 \
  -v ogmemory-gpu_hf-cache:/root/.cache/huggingface \
  ogmemory-omni:local -c \
  'from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen3-Omni-30B-A3B-Instruct", revision="26291f793822fb6be9555850f06dfe95f2d7e695", max_workers=8)'
docker run --rm --entrypoint /opt/qwen3/bin/python \
  -v ogmemory-gpu_hf-cache:/root/.cache/huggingface -v "$PWD:/work" \
  ogmemory-qwen3-trial:local /work/scripts/verify_qwen3_snapshot.py
```

Startup smoke requires three files under `results/qwen3-omni-20261022/media`:
`cars.jpg`, `living_room.jpg`, `living_room.wav`. The latter two come from
`/data/m3_data/videos/robot/living_room_22.mp4`, at 30 seconds, with 10 seconds
of mono 16 kHz audio. Example recreation:

```bash
mkdir -p results/qwen3-omni-20261022/media
curl -fL https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cars.jpg \
  -o results/qwen3-omni-20261022/media/cars.jpg
ffmpeg -ss 30 -i /data/m3_data/videos/robot/living_room_22.mp4 \
  -frames:v 1 -vf scale=448:-2 -y results/qwen3-omni-20261022/media/living_room.jpg
ffmpeg -ss 30 -i /data/m3_data/videos/robot/living_room_22.mp4 \
  -t 10 -vn -ar 16000 -ac 1 -y results/qwen3-omni-20261022/media/living_room.wav
```

**Starting the trial requires pausing the existing GPU workloads.** Do not run
this beside the old Omni/text deployment without a separate memory-budget test:

```bash
docker stop t4-omni-omni-1 t4-omni-text-1 t4-omni-embeddings-1 t4-omni-cv-gpu-1
docker compose -f compose.qwen3-trial.yaml up -d --no-build
docker compose -f compose.qwen3-trial.yaml logs -f
```

Loading plus startup smoke takes several minutes. Once healthy:

- Base address: `http://127.0.0.1:8003/v1`.
- Model ID: `qwen3-omni-30b-a3b-nf4`.
- Bearer key: the existing `.env` `API_KEY`.
- Routes: `/health`, `/v1/models`, `/v1/chat/completions`.
- Supports string messages, image data URIs, base64 `input_audio`, text responses.
- Temperature zero selects deterministic generation; positive temperature enables
  sampling. Native tools and post-validated JSON object mode are supported as
  described above. No streaming, grammar-constrained JSON, speech output, video
  API, or concurrent inference support.
- Request guard: input + requested output ≤ 32768; requested output ≤ 8192.
  **This is a configured guard, not proof that every 32K multimodal workload fits.**

To restore the original deployment:

```bash
docker compose -f compose.qwen3-trial.yaml stop
docker start t4-omni-embeddings-1 t4-omni-cv-gpu-1
# Wait for the embedding service health check before starting the text model.
until curl -fsS http://127.0.0.1:8001/health >/dev/null; do sleep 3; done
docker start t4-omni-omni-1 t4-omni-text-1
```

The initial deployment experiment does not modify oGMemory, LoCoMo configs, or database contents.
The original services are restored after validation; the new model is retained
as an on-demand trial rather than silently replacing the existing interfaces.
Full 30B M3 ingestion/LoCoMo integration, throughput optimization, and quality
acceptance remain separate work. Set `QWEN3_TRIAL_RESULT_DIR` to a container-side
output directory when starting a new trial, and populate its `media/` directory,
so its startup evidence does not overwrite the original experiment.

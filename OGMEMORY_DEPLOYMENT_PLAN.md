# Local GPU deployment plan for oGMemory / M3

Date: 2026-10-16. Status: inspected, not deployed or GPU validated.

## Decision

Use Compose to manage a **small Omni endpoint, a text embedding endpoint, and
specialist face/audio workers**. Do not ask the Omni model to replace face
identity vectors, speaker vectors, or timestamped diarization.

Start with Qwen/Qwen2.5-Omni-3B as a compatibility/throughput candidate and
compare Qwen/Qwen2.5-Omni-7B after the real request shape passes. Use
BAAI/bge-m3 for dense text retrieval. Preserve the existing InsightFace
buffalo_l, Whisper large-v3-turbo, pyannote speaker-diarization-3.1, and
SpeechBrain ECAPA models initially so model migration does not confound
the benchmark.

These are **provisional candidates, not a latest-model survey**. Web search
and direct official model-card retrieval produced no usable content in this
session; direct HTTPS failed DNS resolution. Recheck official model cards,
licenses, serving support, and image tags before downloading or deployment.
In particular, do not assume Omni-3B and Omni-7B have identical licenses.

## Observed environment and application

- PCI inventory exposes four Tesla T4 devices. Actual usable VRAM, driver,
  GPU load and topology could not be measured: `nvidia-smi` cannot communicate
  with the driver in this session. This does not establish whether the host
  driver is broken or simply inaccessible from the execution environment.
- Host reports 125 GiB RAM and approximately 941 GiB free workspace disk.
- Compose CLI is present (v5.6.0). Docker socket access is denied, so existing
  container state and runtime networking could not be inspected.
- No running services or oGMemory files were changed. Its repository is
  outside this session's writable roots. Git metadata in the deployment
  workspace is unavailable; preserve existing files rather than replace them.
- Existing deployment files are a one-image, 4K-context, single-sequence
  baseline, not an M3-scale configuration.

Source inspection under `/home/yuanjian/Development/memory/oGMemory`:

| Source | Consequence |
|---|---|
| `config/m3/pipelines/video_episodic_full.yaml` | 30-second clips, up to 16 frames at 512px, audio, ASR, diarization, speaker embeddings, faces, indexing, consolidation and QA |
| `extraction/operators/builtin/episodic_memory_generator.py` | Sends multiple `image_url` parts, optional `input_audio`, and transcript text; needs structured episodic output |
| `providers/llm/openai_llm.py` | OpenAI-style chat interface; validate JSON handling and QA tool calls, not just text responses |
| `m3/config.py`, `m3/llm_factory.py` | Separate `m3.episodic_llm`, `m3.qa_llm`, `m3.judge_llm`, and `m3.embedding` overrides; top-level LLM remains relevant |
| `deploy/docker-compose.yml` | oGMemory uses host networking; local model ports on host loopback are appropriate for this topology |
| `deploy/docker-compose.cv.yml`, `docker/Dockerfile.cv`, `scripts/cv_service.py` | Current CV is CPU InsightFace; constructor accepts providers, but CLI does not expose GPU selection; object and pose routes are stubs |
| `m3/pipeline/stages.py` | ASR, diarization and speaker embedding stages forward `device`; defaults in their CLIs/operators are CPU |
| `scripts/run_m3_bench_subset.py` | Existing resumable multi-video subprocess runner; use it before designing a new distributed scheduler |

The existing README's statement that the M3 repository was not supplied is
superseded by this inspection.

## Model shortlist

The following suitability judgments are planning hypotheses to validate.
There are no measured quality, VRAM, or throughput results yet.

| Role | Initial candidate | Alternative / decision gate |
|---|---|---|
| Native image + audio → episodic text | Qwen2.5-Omni-3B | Lowest-cost baseline already configured here; license and mixed-media serving must pass |
| Higher-quality native Omni | Qwen2.5-Omni-7B | Compare on two T4s; TP compatibility and PCIe cost must be measured |
| Image + transcript ablation | Qwen2.5-VL-3B-Instruct or 7B-Instruct | Not an audio-native substitute; run ASR and disable raw audio attachment explicitly |
| Newer large Omni candidate | Qwen3-Omni-30B-A3B family | Research-only for this host; verify exact current variants/support. Sparse active parameters do not imply only those weights reside in memory |
| Dense text retrieval | BAAI/bge-m3, 1024 dimensions | Keep the current embedder as a control. Consider Qwen3-Embedding-0.6B only as a separate retrieval experiment with verified serving/pooling/instruction requirements |
| Faces and face embeddings | Existing InsightFace buffalo_l | GPU ONNX Runtime build, explicit CUDA provider, fail if CUDA is unavailable |
| Timestamped speech | Existing faster-whisper large-v3-turbo | GPU worker; test CUDA compute type supported by its pinned build |
| Speaker diarization | Existing pyannote/speaker-diarization-3.1 | Requires authorized model access; keep distinct from speech recognition |
| Speaker embeddings | Existing speechbrain/spkrec-ecapa-voxceleb | Preserve existing 192-dimensional contract |
| QA / semantic consolidation | Reuse Omni initially | If tool-call/schema tests fail, evaluate a separate text instruction model such as Qwen2.5-7B-Instruct; provision it in a mutually exclusive phase |
| Judge | Fixed separate configuration | A local self-judge is only a smoke metric; retain fixed human-labeled or independent reference evaluation for quality claims |

No speech generation service is needed. BGE-M3 embeds extracted text, not raw
images/audio. True cross-modal vector retrieval would require a distinct
encoder, API integration and index experiment; it is not part of this migration.
Do not introduce object/pose models until their stub API implementations and
actual benchmark requirements are addressed.

## GPU allocation and Compose layout

Treat four cards as four independent memory budgets, not one 64 GB device.
Nominal T4 capacity must be confirmed with `nvidia-smi`.

### Bootstrap: correctness and measurement

| GPU | Work |
|---|---|
| 0 | Omni-3B, FP16, single active request |
| 1 | BGE-M3 embedding service |
| 2 | GPU face CV |
| 3 | ASR → diarization → speaker embeddings, serial stages initially |

### Quality comparison: mutually exclusive with bootstrap allocation

| GPU | Work |
|---|---|
| 0–1 | Omni-7B TP=2 |
| 2 | Embeddings; optionally CV only after measuring combined peak allocation |
| 3 | Audio workers |

Precompute CV/ASR if needed instead of overcommitting a shared card. GPU
device reservations select devices; they do not impose VRAM quotas.

### Throughput mode: only after profiling

Try two independent 3B replicas on GPUs 0 and 1, embeddings plus measured/
scheduled CV on GPU 2, audio on GPU 3. Route jobs explicitly or through a
bounded-queue gateway. Prefer measured replicas over assuming TP accelerates
small models. If sharing GPU 2 is unstable, precompute CV and run it in a
separate phase. Do not run quality and throughput allocations together.

Deployment layout update (2026-10-19): the only active entry point is the
root `compose.yaml`. The former nested stack has been consolidated there.
This planning document's remaining worker/quality ideas are not implemented
services; use the root README for actual startup commands. Service scope:

- `omni`: OpenAI-compatible chat, host loopback port 8000.
- `embeddings`: OpenAI-compatible embeddings, host loopback port 8001.
- `cv-gpu`: existing face API contract, separate test port 18081 initially.
- `m3-worker`: GPU-enabled oGMemory pipeline image with pinned ASR/diarization/
  speaker dependencies; read-only dataset mounts and writable run/cache mounts.
- Optional quality/replica services behind explicit profiles. Profiles are
  not mutual-exclusion locks: preflight must reject overlapping GPU assignments.

Keep the model Compose project separate from existing databases. Do not
recreate databases, prune networks, or use `down -v`. Bind unauthenticated CV
only to loopback or a private bridge. Host-networked oGMemory can use the
published loopback endpoints; a bridge-networked worker needs reachable host
addresses or shared service networking, not its own localhost.

Use named model caches, GPU UUID assignments, log rotation, health checks,
readiness probes, pinned image digests and model commits. Do not put real keys
in committed YAML or benchmark manifests. Keep startup failures visible;
enable restart policy only after readiness is reliable.

## Compatibility work before a full benchmark

1. **T4 serving stack:** the existing vLLM 0.10.2/V0/XFormers and TEI
   `turing-1.8` settings are candidates, not verified pins. Test exact image
   availability, CUDA tensor allocation and model-specific kernels. Do not
   blindly upgrade to `latest`, assume BF16/FlashAttention support, or add
   quantization before establishing a working FP16 baseline.
2. **Request limits:** existing `image:1` rejects the full pipeline's up to
   16 images. Raise the limit in a separate full-workload profile; retain one
   audio item. The current pipeline sends frames, so native `video_url`
   support is not required.
3. **Context:** 4096 total tokens cannot coexist with a 4096-token output
   budget plus nonempty input. Set an explicit smaller output budget for
   smoke tests. Profile 8K/16K total context candidates with actual 16-frame,
   30-second-audio payloads; neither fits by assertion. Include image/audio
   tokens, transcript, schema and output allowance in admission checks.
4. **Memory admission:** bound frame count/resolution, audio duration, input
   tokens, output tokens, active requests and queue length. Start at one
   active request. Reduced-frame/short-clip runs are smoke tests or labeled
   ablations, not silently equivalent benchmark results.
5. **API semantics:** verify mixed image/audio requests, episodic schema,
   JSON fallback, reasoning-output cleanup, QA tool calls and timeouts.
   OpenAI-style URL compatibility alone is insufficient.
6. **CV GPU:** build a CUDA-compatible ONNX Runtime image, avoid conflicting
   CPU/GPU runtime packages, expose provider selection in the service,
   and fail startup if CUDA cannot be used. Health should report actual
   session providers; real face requests must return finite 512-D vectors.
7. **Audio GPU:** set the three stage devices explicitly to CUDA. Initially
   run them sequentially; check process/model lifetime before concurrent
   jobs. Confirm authorized pyannote access and propagate failures rather
   than counting fallback spectral fingerprints as a successful full run.

## oGMemory configuration mapping

Prepare a separate local-test copy of the current resolved config, preserving
storage and vector DB connection strings. Do not overwrite live config.

| Actual setting | Proposed local value |
|---|---|
| `llm.provider` | `openai` |
| `llm.base_url` | `http://127.0.0.1:8000/v1` for host-networked processes |
| `llm.model` | Served alias `omni-3b` |
| `llm.api_key` | Local server key |
| `m3.episodic_llm` | Explicit local provider/base URL/model/key |
| `m3.qa_llm` | Explicit tested local QA endpoint |
| `m3.judge_llm` | Explicit local judge for fully offline testing, or deliberately retained external judge for a separately labeled control |
| `embedding` and `m3.embedding` | Local embedding endpoint, `BAAI/bge-m3`, local key; use ordinary text embeddings, not the cloud multimodal endpoint |
| `m3.embedding.dimension`, `vector_db.dimension` | 1024 for BGE-M3 |
| `m3.vector_index.table_name` | New experiment-specific table |
| `OGMEM_CV_SERVICE_URL` | `http://127.0.0.1:18081` during GPU-CV validation |

Inspect all resolved role overrides: changing `LLM_BASE_URL` alone does not
replace explicit YAML endpoints. Verify no unintended external calls remain.
The M3 vector index inherits its DSN from `vector_db.connection_string`;
graph storage falls back to `storage.connection_string`, not the vector DSN.

Never mix new vectors with old vectors even if dimensions match. Use isolated
test tables/collections and graph/run namespaces, rebuild embeddings, and
retain old artifacts for rollback. Also use distinct run/cache directories
for model, prompt, frame or audio changes: the subset runner's resume logic
must not reuse an answer simply because its question ID already exists.

## Execution stages and acceptance gates

1. **Host gate:** verify driver/GPU access from the actual host, Docker
   permissions, container toolkit, topology, free VRAM and registry/model
   access. Run CUDA in each exact candidate image. Do not install drivers
   or change host services based solely on sandbox failures.
2. **Endpoint gate:** start only small Omni + embeddings. Test text,
   image-only, audio-only, mixed media, embedding batches and malformed/
   over-limit requests. Verify real structured output, not HTTP 200 alone.
3. **Pipeline gate:** one annotated short video in an isolated run. Test
   full 16-frame/30-second requests as a separate gate before scaling.
   Check provenance, nonempty episodic records, valid face/speaker vectors,
   transcript coverage, index counts, scoped retrieval and QA completion.
4. **Specialist gate:** move CV and audio to GPU one at a time. Confirm actual
   GPU execution and compare against the same CPU models on fixed samples.
5. **Pilot gate:** fixed representative 10-video set: speech/no speech,
   multiple speakers, visible/no faces, difficult lighting and longer clips.
   Compare 3B versus 7B and optional VL+transcript ablation. Keep retrieval,
   CV, judge, prompts and annotations fixed where the experiment permits.
6. **Scale gate:** sweep worker concurrency 1 → 2 → 4 under bounded queues,
   then run a sustained representative batch. Measure stage latency p50/p95,
   clips/minute, videos/hour, tokens/second, peak VRAM, GPU/CPU utilization,
   queue wait, retries, OOMs, schema failures and QA quality. Choose capacity
   from measured bottlenecks rather than assigning a target without data.
7. **Reproducibility:** store dataset IDs, seed, model revisions, image
   digests, code revision when available, full nonsecret config, modality/
   sampling policy and metric definitions. Resume and retry tests must
   demonstrate no duplicate/cross-run indexing. Treat skipped stages and
   partial QA results as incomplete runs.

Success criteria: all required stages execute without unintended fallback,
no unexpected external model calls, no OOMs at selected sustained concurrency,
valid retrieval dimensions/scoping, and acceptable accuracy against an agreed
fixed control. Set the accuracy tolerance after the pilot, before full scale.

## Next implementation deliverables

1. Host-access confirmation and verified model/license/runtime selection.
2. Separate Compose file and local env template with GPU-allocation preflight.
3. oGMemory local config template and isolated experiment pipeline settings.
4. GPU CV provider support/image and GPU M3 worker image (oGMemory changes
   require a writable session for that repository).
5. Full-payload integration smoke tests and repeatable load-test/report script.

Official primary-source lookup targets for the follow-up (not successfully
retrieved or verified during this session):

```text
https://huggingface.co/Qwen/Qwen2.5-Omni-3B
https://huggingface.co/Qwen/Qwen2.5-Omni-7B
https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct
https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
https://huggingface.co/BAAI/bge-m3
https://docs.vllm.ai/
https://huggingface.co/docs/text-embeddings-inference/
https://github.com/deepinsight/insightface
https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
https://github.com/SYSTRAN/faster-whisper
https://huggingface.co/pyannote/speaker-diarization-3.1
https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
```

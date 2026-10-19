# Small Omni + embeddings on four Tesla T4 GPUs

## Recommendation and scope

Start with **Qwen2.5-Omni-3B on one T4 and BGE-M3 on another**. After that
passes representative multimodal requests, optionally use the remaining two
cards for **Qwen2.5-Omni-7B**, tensor-parallel across two GPUs. This provides a
small baseline and a larger comparison without stopping the embedding service.

This is a deployment candidate, **not a GPU-validated configuration**.
In the preparation environment, `nvidia-smi` could not communicate with the
driver and outbound DNS failed. Image availability, container CLI compatibility,
CUDA execution, model loading and VRAM fit must therefore pass the steps below
on the actual GPU host. The supplied versions are conservative compatibility
candidates, not claims about the latest releases or production security support.

**Which M3?** The benchmark repository/paper was not supplied. BGE-M3 is an
embedding model; it is not itself the requested benchmark. These files provide
generic OpenAI-style chat and embedding endpoints, not an M3 runner, dataset,
scoring implementation or a guarantee of compatibility with a particular M3.
Provide the benchmark URL and required modalities to finalize the adapter.

### Proposed GPU allocation

| GPU | Service | Model | Host endpoint |
|---|---|---|---|
| 0 | `omni` | Qwen/Qwen2.5-Omni-3B | `http://127.0.0.1:8000/v1` |
| 1 | `embeddings` | BAAI/bge-m3 | `http://127.0.0.1:8001/v1` |
| 2–3 | optional `omni-7b` | Qwen/Qwen2.5-Omni-7B, TP=2 | `http://127.0.0.1:8002/v1` |

Assumes nominal 16 GB T4 cards, with no other GPU consumers. Four separate cards
are not one pooled 64 GB GPU. Parameter-count-only FP16 weight estimates are
roughly 6 GB for 3B and 14 GB for 7B; these **exclude** multimodal encoders,
activations, KV cache, CUDA/runtime allocations and loading peaks. They are not
memory guarantees. The embedding model is much smaller; a dedicated card keeps
its memory and latency independent during benchmarking.

T4 is a Turing GPU. This configuration avoids BF16 and FlashAttention-2:

- Omni uses FP16, vLLM's legacy V0 engine, XFormers attention, eager execution,
  one sequence at a time, and a 4,096-token total context budget.
- The embedding service uses the separate TEI `turing` image, FP16 and
  `USE_FLASH_ATTENTION=False`. Turing support has historically been experimental;
  validate the selected tag before relying on it.
- The 7B service disables custom all-reduce and uses ordinary NCCL collectives.
  Two-card PCIe communication can make it slower; do not infer speedup from TP.
- These settings still need an actual multimodal startup test: selecting an
  attention backend does not prove every model-specific encoder kernel works.

### What these services do / do not do

- **Omni:** text, image and audio input with **text output** through vLLM's
  Thinker path. This is not a speech synthesis or streaming speech deployment.
  The default allows one image and one audio item per request. Test short WAV/MP3
  clips first (for example, 5–10 seconds) and small images.
- **Video:** off by default (`VIDEO_LIMIT=0`) to constrain profiling and memory.
  Set `VIDEO_LIMIT=1` and recreate the Omni service only if your benchmark needs
  video. Validate your pinned vLLM video request format and explicitly bound
  duration/frame sampling in the benchmark client. Image pixel limits do not
  bound audio duration or video frame count. Video is not covered by the script.
- **BGE-M3:** dense, normalized, 1,024-dimensional **text** vectors with CLS
  pooling. The model has an 8,192-token input limit; limits are tokenizer tokens,
  not characters. No silent truncation is enabled here. TEI batching is deliberately
  conservative. This endpoint does not expose BGE-M3 sparse or ColBERT outputs,
  nor joint image/audio/video embeddings.
- If M3 requires multimodal retrieval vectors, speech output, long video or
  longer contexts, **this baseline is insufficient without changes**.
- Check the exact model revision's license and use restrictions before use.
  In particular, treat the Omni-3B research-license terms separately from the
  7B model's terms; this repository does not grant rights to any weights.

## Deployment plan

### 1. Verify the host

Install a compatible NVIDIA driver, Docker Engine, Docker Compose with GPU
device reservations, and NVIDIA Container Toolkit using their official host
installation procedures. Do not install drivers inside these containers.
The driver must support the CUDA runtime in **both selected images**. A CUDA
toolkit installed on the host is not a replacement for a functioning driver.

Budget roughly 100 GB of free disk for images, cached models and headroom,
and preferably at least 32 GB host RAM (64 GB or more for loading all services).
These are planning budgets, not measured minima. Confirm internet access to
Docker Hub, GHCR and Hugging Face, or preload images and model caches offline.

```bash
cp .env.example .env
python3 -c 'import secrets; print(secrets.token_hex(32))'
# Put the generated value in .env as API_KEY; optionally set HF_TOKEN.
chmod 600 .env
bash scripts/preflight.sh
```

The preflight intentionally fails if Docker or host GPU access is unavailable.
It does not fix or alter the host. Run it in a shell with Docker permissions.
Inspect `nvidia-smi topo -m` when choosing the two 7B GPUs.

### 2. Pull and verify the actual serving images

```bash
docker compose pull omni embeddings
docker compose run --rm --no-deps --entrypoint python3 omni -c \
  'import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0)); print(torch.ones(1, device="cuda"))'
docker compose run --rm --no-deps omni --help
docker compose run --rm --no-deps embeddings --help
```

Check for a T4 with capability `(7, 5)`, a successfully allocated CUDA tensor,
and support for the supplied CLI options. If a selected image tag is unavailable
or its options differ, stop and inspect the official version documentation;
do not silently substitute `latest`. Newer vLLM releases may not retain the
legacy engine/backend combination this plan uses.

### 3. Start the small baseline

```bash
docker compose up -d omni embeddings
docker compose logs -f --tail=100 omni embeddings
```

Initial downloads and startup profiling may take a long time. The Omni service
has a long startup health-check grace period. TEI readiness is checked by the
real embedding request below, not by an image-dependent `curl` health check.
Restart is disabled intentionally: a failed model load should remain visible,
not loop repeatedly.

In another shell, export only the key needed by the client:

```bash
export API_KEY='the-same-key-you-put-in-.env'
python3 scripts/smoke_test.py
python3 scripts/smoke_test.py --image /path/to/small.jpg --audio /path/to/short.wav
```

The first call verifies text output and vector shape/normalization. The second
also sends image-only, audio-only and combined image/audio requests. Nonempty
responses are not proof of correct perception: inspect them, then run your
benchmark's own known-answer samples. `.env` is read by Compose; Python does
**not** automatically load it.

### 4. Add the optional 7B comparison

```bash
docker compose --profile compare pull omni-7b
docker compose --profile compare up -d omni-7b
docker compose logs -f --tail=100 omni-7b
python3 scripts/smoke_test.py \
  --omni-url http://127.0.0.1:8002 --model omni-7b \
  --image /path/to/small.jpg --audio /path/to/short.wav
```

Watch `nvidia-smi` during startup and representative longest requests. Start
benchmark concurrency at **one**; increase only after measuring peak memory
and latency. Do not run the two comparison benchmarks concurrently when
collecting clean latency measurements (they still share CPU, disk and PCIe).

### 5. Wire up the actual M3 benchmark

Map the benchmark's configuration to these values; **these are connection
settings, not asserted M3 configuration key names**:

| Setting | Small baseline | Larger comparison |
|---|---|---|
| Chat base URL | `http://127.0.0.1:8000/v1` | `http://127.0.0.1:8002/v1` |
| Chat model name | `omni-3b` | `omni-7b` |
| Embedding base URL | `http://127.0.0.1:8001/v1` | same |
| Embedding model name | `BAAI/bge-m3` | same |
| API key | value from `.env` | same |
| Embedding dimension | 1024 | 1024 |

If the benchmark is a container attached to the Compose network, use
`http://omni:8000/v1`, `http://omni-7b:8000/v1`, and
`http://embeddings:80/v1` instead; `localhost` inside that container is itself.

Before a scored run:

1. Obtain the exact benchmark repository/commit, dataset/version, license,
   evaluation split, prompts, metrics, and required input/output modalities.
2. Check whether the embedding model is prescribed. If it is, replacing it
   with BGE-M3 changes the experiment. Rebuild indexes when changing models.
3. Verify context lengths, media limits, truncation policy and audio/video
   synchronization requirements. A 4K context includes input and generated
   tokens, not just the text question.
4. Pin Hugging Face commit IDs in the `*_REVISION` variables and replace image
   tags with tested digests. Recreate services after pinning.
5. Record GPU/driver information, image digests, model revisions, benchmark
   commit, sampling settings and media preprocessing. Record cold-start and
   warmed-up performance separately; record latency, throughput and peak VRAM.
6. Run a tiny correctness subset before the complete benchmark. Save raw
   responses alongside scoring results so format failures remain visible.

Do not publish `docker compose config` output unredacted: it contains secrets.

## Operations and troubleshooting

- Host `nvidia-smi` failure: repair host GPU visibility/driver first, or repeat
  from the real host if this shell is an unprivileged container.
- Host GPUs work but container CUDA fails: check NVIDIA Container Toolkit,
  Docker runtime configuration and driver/runtime compatibility.
- BF16 / FlashAttention errors: verify FP16, V0 and XFormers settings actually
  took effect. Encoder-specific failures may require another tested version;
  do not assume changing only the text attention setting fixes them.
- Omni out of memory: keep concurrency at one, reduce media size/duration,
  disable video and try `OMNI_CONTEXT=2048`. Reducing the memory utilization
  fraction mainly reduces the cache budget; it cannot make model weights fit.
  If 3B still cannot fit, allocate two free GPUs to its device reservation and
  add `--tensor-parallel-size 2` / `--disable-custom-all-reduce`, with the 7B
  profile stopped. That is a manual alternative, not an included auto-fallback.
- TEI errors/OOM: inspect startup and token-length errors, check CLS pooling,
  keep small batches, and validate the Turing image. Do not silently truncate
  benchmark inputs to make requests pass.
- NCCL failure on 7B: check that both assigned cards are visible and inspect
  the PCIe topology and logs before adjusting NCCL settings.
- `docker compose --profile compare down` stops/removes all services but
  preserves cached models. Adding `-v` deletes the named caches.

Ports bind only to loopback and services require an API key. For remote access,
use an SSH tunnel rather than exposing plain HTTP; terminate TLS and add access
controls if deliberately serving beyond localhost. Images run third-party code:
review and scan the selected versions before production use.

## Local validation

These commands need no GPU and do not download models:

```bash
docker compose --env-file .env.example --profile compare config --quiet
bash -n scripts/preflight.sh
python3 -m unittest discover -s tests -v
python3 scripts/smoke_test.py --help
```

They validate YAML interpolation and local helper behavior only. Successful
image pulls, startup, endpoint smoke tests and an actual M3 run remain separate
acceptance gates.

## Official references to check on the deployment host

Online verification was unavailable during preparation. These are reference
locations, not evidence that the supplied image/model combination was tested:

```text
https://huggingface.co/Qwen/Qwen2.5-Omni-3B
https://huggingface.co/Qwen/Qwen2.5-Omni-7B
https://github.com/QwenLM/Qwen2.5-Omni
https://huggingface.co/BAAI/bge-m3
https://docs.vllm.ai/en/v0.10.2/
https://github.com/huggingface/text-embeddings-inference
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
https://docs.docker.com/compose/how-tos/gpu-support/
```

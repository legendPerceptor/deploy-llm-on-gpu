"""Verify every official safetensors shard against its Hugging Face LFS digest."""
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from qwen3_omni_trial import MODEL, REVISION

root = Path(snapshot_download(MODEL, revision=REVISION, local_files_only=True))
info = HfApi().model_info(MODEL, revision=REVISION, files_metadata=True)
records = []
for entry in info.siblings:
    if not entry.rfilename.endswith(".safetensors"):
        continue
    path = root / entry.rfilename
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    record = {"file": entry.rfilename, "bytes": path.stat().st_size,
              "expected_sha256": entry.lfs.sha256, "actual_sha256": digest.hexdigest()}
    record["ok"] = record["actual_sha256"] == record["expected_sha256"] and record["bytes"] == entry.size
    records.append(record)
    print(json.dumps(record), flush=True)
output = Path("/work/results/qwen3-omni-20261022/sha256.json")
output.write_text(json.dumps({"revision": REVISION, "files": records}, indent=2))
assert len(records) == 15 and all(r["ok"] for r in records), "Incomplete or corrupt checkpoint"

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v docker >/dev/null
command -v nvidia-smi >/dev/null
docker compose version
docker info >/dev/null
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv
count="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l)"
if (( count < 4 )); then
  echo "Expected at least four GPUs; found $count." >&2
  exit 1
fi
nvidia-smi topo -m
docker compose --profile compare config --quiet
echo "Host checks passed. Next test CUDA inside the exact serving image:"
echo "docker compose run --rm --no-deps --entrypoint python3 omni -c 'import torch; print(torch.cuda.get_device_name(0)); print(torch.ones(1, device=\"cuda\"))'"
echo "This script does not install drivers or change host configuration."

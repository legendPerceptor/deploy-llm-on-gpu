#!/usr/bin/env python3
"""Read-only host and Compose checks. Does not start or stop containers."""
import csv
import io
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True)


def validate_allocation(config, inventory):
    lookup = {}
    for row in inventory:
        index, uuid, name, total, used = [item.strip() for item in row]
        lookup[index] = lookup[uuid] = (uuid, name, total, used)
    seen = {}
    for service, spec in config["services"].items():
        devices = spec["deploy"]["resources"]["reservations"]["devices"]
        for device in devices:
            for identifier in device["device_ids"]:
                if identifier not in lookup:
                    raise ValueError(f"{service}: unknown GPU {identifier}")
                uuid, name, total, used = lookup[identifier]
                if uuid in seen:
                    raise ValueError(f"{service} overlaps {seen[uuid]} on {uuid}")
                seen[uuid] = service
    return seen


def main():
    config = json.loads(run("docker", "compose", "config", "--format", "json"))
    command = config["services"]["omni"]["command"]
    key = command[command.index("--api-key") + 1]
    if len(key) < 32 or key == "replace-with-a-random-local-key":
        raise ValueError("Set a random API_KEY of at least 32 characters")
    run("docker", "info")
    output = run(
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used",
        "--format=csv,noheader,nounits",
    )
    print(output)
    allocation = validate_allocation(config, list(csv.reader(io.StringIO(output))))
    print(json.dumps(allocation, indent=2))
    print("Host/config checks passed; free VRAM, CUDA kernels and endpoints still need validation.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, subprocess.CalledProcessError, ValueError, KeyError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)

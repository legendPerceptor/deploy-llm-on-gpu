import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DeploymentTests(unittest.TestCase):
    def test_single_root_entrypoint(self):
        project = ROOT.parent
        self.assertTrue((project / "compose.yaml").is_file())
        self.assertFalse((project / "ogmemory-gpu").exists())
        self.assertEqual(load("preflight").ROOT, project)
        for name in ("Dockerfile.cv", "Dockerfile.omni", "requirements-cv.txt"):
            self.assertTrue((project / name).is_file())
        self.assertIn("COPY scripts/cv_gpu.py /app/cv_gpu.py",
                      (project / "Dockerfile.cv").read_text())

    @unittest.skipUnless(shutil.which("docker"), "Docker CLI is not installed")
    def test_compose_layout(self):
        project = ROOT.parent
        config = json.loads(subprocess.check_output([
            "docker", "compose", "--project-directory", str(project),
            "--env-file", str(project / ".env.example"),
            "-f", str(project / "compose.yaml"), "config", "--format", "json",
        ], cwd="/", text=True))
        self.assertEqual(config["name"], "t4-omni")
        self.assertEqual(set(config["services"]), {"omni", "embeddings", "cv-gpu"})
        for name in ("hf-cache", "embedding-cache", "face-cache"):
            self.assertEqual(config["volumes"][name]["name"], "ogmemory-gpu_" + name)
        for service in ("omni", "cv-gpu"):
            build = config["services"][service]["build"]
            self.assertEqual(Path(build["context"]), project)
            self.assertTrue((project / build["dockerfile"]).is_file())
        self.assertEqual(config["services"]["omni"]["image"], "ogmemory-omni:local")
        self.assertEqual(config["services"]["cv-gpu"]["image"], "ogmemory-gpu-cv-gpu:latest")

    def test_cuda_guard(self):
        check = load("cv_gpu").validate_sessions
        def model(providers):
            return SimpleNamespace(session=SimpleNamespace(get_providers=lambda: providers))
        good = model(["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.assertEqual(len(check({"detection": good, "recognition": good})), 2)
        with self.assertRaises(RuntimeError):
            check({"detection": good})
        with self.assertRaises(RuntimeError):
            check({"detection": good, "recognition": model(["CPUExecutionProvider"])})

    def test_gpu_alias_overlap(self):
        check = load("preflight").validate_allocation
        def service(device):
            return {"deploy": {"resources": {"reservations": {
                "devices": [{"device_ids": [device]}]
            }}}}
        inventory = [["0", "GPU-a", "Tesla T4", "15360", "0"]]
        self.assertEqual(check({"services": {"one": service("0")}}, inventory), {"GPU-a": "one"})
        with self.assertRaises(ValueError):
            check({"services": {"one": service("0"), "two": service("GPU-a")}}, inventory)
        with self.assertRaises(ValueError):
            check({"services": {"one": service("9")}}, inventory)

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1] / "ogmemory-gpu"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DeploymentTests(unittest.TestCase):
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

"""Tests for the lightweight installation verification entry point."""

from __future__ import annotations

import importlib.util
import json
import socket
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "tools" / "check_install.py"
SPEC = importlib.util.spec_from_file_location("check_install", SCRIPT_PATH)
assert SPEC and SPEC.loader
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


def test_optional_groups_are_non_importing_and_match_public_extras():
    available = {
        "FrEIA",
        "networkx",
        "mpmath",
        "skimage",
        "matplotlib",
    }
    calls = []

    def find_spec(module_name):
        calls.append(module_name)
        return object() if module_name in available else None

    groups = {
        item["group"]: item
        for item in CHECKER.inspect_optional_groups(find_spec=find_spec)
    }

    assert set(groups) == {
        "dev",
        "flow",
        "vl",
        "saa",
        "geomloss",
        "glass",
        "visualization",
        "faiss-cpu",
        "all",
    }
    assert groups["flow"]["available"] is True
    assert groups["visualization"]["available"] is True
    assert groups["vl"]["available"] is False
    assert groups["all"]["available"] is False
    assert "open_clip" in groups["all"]["missing_modules"]
    assert set(calls) >= available


def test_core_version_check_never_queries_gpu_state():
    class ForbiddenGPU:
        def __getattr__(self, name):
            raise AssertionError(f"GPU state was queried: {name}")

    modules = {
        "baoiad": SimpleNamespace(__version__="0.1.0"),
        "torch": SimpleNamespace(__version__="2.0.0", cuda=ForbiddenGPU()),
        "mmengine": SimpleNamespace(__version__="0.10.0"),
        "mmcv": SimpleNamespace(__version__="2.0.0"),
    }

    checks = CHECKER.inspect_core_packages(import_module=modules.__getitem__)

    assert all(item["available"] for item in checks)
    assert {item["name"] for item in checks} == {
        "BaoIAD",
        "PyTorch",
        "MMEngine",
        "MMCV",
    }


def test_resolved_paths_do_not_require_dataset_or_cache_directories(tmp_path):
    environment = {
        "HOME": str(tmp_path / "home"),
        "BAOIAD_DATA_ROOT": "private-data",
        "HF_HOME": "relative-hf",
        "TORCH_HOME": str(tmp_path / "torch"),
        "BAOIAD_CACHE_DIR": "relative-baoiad-cache",
    }

    paths, errors = CHECKER.resolve_paths(environment)

    assert errors == []
    assert paths["data_root"] == str(CHECKER.ROOT / "private-data")
    assert paths["hf_home"] == str(Path.cwd() / "relative-hf")
    assert paths["torch_home"] == str(tmp_path / "torch")
    assert paths["baoiad_cache_dir"] == str(Path.cwd() / "relative-baoiad-cache")
    assert not (CHECKER.ROOT / "private-data").exists()


def test_cpu_only_offline_check_is_network_free_and_exits_zero(
    monkeypatch, capsys, tmp_path
):
    def fail_network(*args, **kwargs):
        raise AssertionError("installation verification attempted network access")

    monkeypatch.setattr(socket.socket, "connect", fail_network)
    monkeypatch.setattr(socket, "create_connection", fail_network)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("PYTORCH_MPS_DISABLE", "1")
    monkeypatch.setenv("BAOIAD_DATA_ROOT", str(tmp_path / "missing-data"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    exit_code = CHECKER.main(["--offline", "--json"])
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert report["ok"] is True
    assert report["runtime_policy"] == {
        "offline": True,
        "network_access": "not attempted",
        "dataset_access": "not attempted",
        "checkpoint_loading": "not attempted",
        "gpu_probe": "not attempted",
    }
    assert all(item["available"] for item in report["core"])
    assert not (tmp_path / "missing-data").exists()
    for name in CHECKER.OFFLINE_ENV_VARS:
        assert CHECKER.os.environ[name] == "1"


def test_missing_optional_groups_do_not_fail_core_verification(monkeypatch):
    core = [
        {
            "name": name,
            "module": module,
            "available": True,
            "version": "1.0",
            "error": None,
        }
        for name, module, _ in CHECKER.CORE_PACKAGES
    ]
    monkeypatch.setattr(CHECKER, "inspect_core_packages", lambda **kwargs: core)

    report = CHECKER.collect_report({}, find_spec=lambda _name: None)

    assert report["ok"] is True
    assert all(not item["available"] for item in report["optional_groups"])

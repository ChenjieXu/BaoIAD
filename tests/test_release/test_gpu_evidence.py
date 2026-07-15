"""Fail-closed tests for the independent real-GPU evidence gate."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / ".github" / "scripts"
CHECKER = SCRIPTS / "check_gpu_evidence.py"
WORKFLOW = ROOT / ".github" / "workflows" / "gpu-smoke.yml"
CHECKOUT_ACTION = "actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5"
UPLOAD_ARTIFACT_ACTION = (
    "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gpu_checker = _load_module("baoiad_gpu_checker", CHECKER)
gpu_runner = _load_module("baoiad_gpu_runner", SCRIPTS / "run_gpu_smoke.py")
_sanitize_log = gpu_runner._sanitize_log
_sanitize_text = gpu_runner._sanitize_text


def _commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return result.stdout.strip()


def _run(evidence: Path) -> tuple[subprocess.CompletedProcess[str], dict]:
    result = subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "--repo-root",
            str(ROOT),
            "--evidence",
            str(evidence),
            "--require-validated",
            "--json",
        ],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result, json.loads(result.stdout)


def test_gpu_workflow_is_manual_self_hosted_and_fail_closed():
    text = WORKFLOW.read_text(encoding="utf-8")
    workflow = yaml.load(text, Loader=yaml.BaseLoader)
    job = workflow["jobs"]["gpu-smoke"]
    run_script = "\n".join(
        step.get("run", "") for step in job["steps"] if isinstance(step, dict)
    )

    assert set(workflow["on"]) == {"workflow_dispatch"}
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["concurrency"]["cancel-in-progress"] == "true"
    assert set(job["runs-on"]) >= {"self-hosted", "linux", "gpu"}
    assert "GPU_EVIDENCE_DIR" not in job["env"]
    assert ".github/scripts/run_gpu_smoke.py" in run_script
    assert ".github/scripts/check_gpu_evidence.py" in run_script
    assert "--require-validated" in run_script
    assert "nvidia-smi --query-gpu=driver_version --format=csv,noheader" in run_script
    assert "python -m pip install --editable ." in run_script
    assert "continue-on-error" not in text
    assert "pull_request_target" not in text
    assert "secrets." not in text

    initialize = next(
        step
        for step in job["steps"]
        if step.get("name") == "Initialize fail-closed evidence"
    )
    initialize_script = initialize["run"]
    assert 'GPU_EVIDENCE_DIR="${RUNNER_TEMP}/baoiad-gpu-evidence"' in (
        initialize_script
    )
    assert "export GPU_EVIDENCE_DIR" in initialize_script
    assert '>> "${GITHUB_ENV}"' in initialize_script
    assert "${{ runner.temp }}" not in initialize_script

    checkout = next(
        step for step in job["steps"] if step.get("uses") == CHECKOUT_ACTION
    )
    assert checkout["with"]["ref"] == "${{ github.sha }}"
    assert checkout["with"]["persist-credentials"] == "false"
    checker = next(
        step
        for step in job["steps"]
        if ".github/scripts/check_gpu_evidence.py" in step.get("run", "")
    )
    assert checker["if"] == "always()"
    upload = next(
        step for step in job["steps"] if step.get("uses") == UPLOAD_ARTIFACT_ACTION
    )
    assert upload["if"] == "always()"
    assert job["steps"].index(checker) < job["steps"].index(upload)


def test_missing_gpu_evidence_is_not_validated(tmp_path):
    result, report = _run(tmp_path / "missing.json")

    assert result.returncode != 0
    assert report["ok"] is False
    assert report["status"] == "not_validated"
    assert "missing" in report["errors"][0]


def test_explicit_not_validated_evidence_fails_require_mode(tmp_path):
    evidence = tmp_path / "gpu-evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "not_validated",
                "reason": "No CUDA device is attached to this runner.",
                "commit_sha": _commit(),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        ),
        encoding="utf-8",
    )

    result, report = _run(evidence)

    assert result.returncode != 0
    assert report == {"errors": [], "ok": False, "status": "not_validated"}


def _fake_validated_document(tmp_path: Path) -> dict:
    log_dir = tmp_path / "logs"
    log_dir.mkdir(exist_ok=True)
    phases = {}
    for method in ("patchcore", "rd", "fastflow"):
        phases[method] = {}
        for phase in ("train", "inference"):
            path = log_dir / f"{method}-{phase}.log"
            path.write_text(f"synthetic {method} {phase}\n", encoding="utf-8")
            phases[method][phase] = {
                "status": "passed",
                "command": f"synthetic-{phase}",
                "duration_seconds": 1.0,
                "peak_vram_bytes": 1024,
                "log_path": f"logs/{path.name}",
                "log_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }

    configs = {
        "patchcore": "configs/patchcore/patchcore_wrn50_256_mvtec_strict.py",
        "rd": "configs/rd/rd_wrn50_256_mvtec_strict.py",
        "fastflow": "configs/fastflow/fastflow_wrn50_256_mvtec_strict.py",
    }
    evidence = tmp_path / "gpu-evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "validated",
                "commit_sha": _commit(),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "runner": "synthetic-test",
                "repository_clean": True,
                "environment": {
                    "cuda_available": True,
                    "python_version": platform.python_version(),
                    "torch_version": "0.0.fake",
                    "torchvision_version": "0.0.fake",
                    "cuda_runtime": "0.0",
                    "mmcv": {
                        "package": "mmcv-lite",
                        "version": "0.0.fake",
                    },
                    "driver_version": "0.0.fake",
                    "device": {
                        "index": 0,
                        "name": "Fake CUDA Device",
                        "compute_capability": "0.0",
                        "total_memory_bytes": 1024,
                    },
                },
                "compiled_cuda_ops": {
                    "status": "available",
                    "checks": [
                        {
                            "name": "torchvision_nms_cuda",
                            "required": True,
                            "available": True,
                            "detail": "synthetic",
                        },
                        {
                            "name": "mmcv_custom_ops",
                            "required": False,
                            "available": False,
                            "detail": "mmcv-lite policy",
                        },
                    ],
                },
                "peak_vram_bytes": 1024,
                "methods": [
                    {
                        "name": method,
                        "config": config,
                        "dataset": "MVTec AD",
                        "category": "bottle",
                        "train": phases[method]["train"],
                        "inference": phases[method]["inference"],
                    }
                    for method, config in configs.items()
                ],
            }
        ),
        encoding="utf-8",
    )
    return json.loads(evidence.read_text(encoding="utf-8"))


def _write_document(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document), encoding="utf-8")


def test_structurally_complete_fake_gpu_evidence_cannot_pass(tmp_path):
    evidence = tmp_path / "gpu-evidence.json"
    _write_document(evidence, _fake_validated_document(tmp_path))

    result, report = _run(evidence)

    assert result.returncode != 0
    assert report["ok"] is False
    assert report["status"] == "not_validated"
    assert not any("private absolute path" in error for error in report["errors"])
    assert any(
        marker in " ".join(report["errors"]).lower()
        for marker in ("cuda", "runtime", "clean exact commit")
    )


def test_validated_evidence_rejects_every_missing_environment_field(tmp_path):
    document = _fake_validated_document(tmp_path)
    required = (
        "cuda_available",
        "python_version",
        "torch_version",
        "torchvision_version",
        "cuda_runtime",
        "mmcv",
        "driver_version",
        "device",
    )

    for field in required:
        candidate = copy.deepcopy(document)
        candidate["environment"].pop(field)
        evidence = tmp_path / f"missing-{field}.json"
        _write_document(evidence, candidate)

        result, report = _run(evidence)

        assert result.returncode != 0
        assert report["status"] == "not_validated"
        assert any(f"environment.{field}" in error for error in report["errors"])


def test_private_paths_and_raw_path_fields_are_rejected(tmp_path):
    document = _fake_validated_document(tmp_path)
    document["dataset_root"] = "/Users/example/private/mvtec_ad"
    document["methods"][0]["train"]["command"] = (
        "python tools/train.py --data /Users/example/private/mvtec_ad"
    )
    document["methods"][1]["inference"]["command"] = (
        r"python tools/test.py --checkpoint C:\private\model.pth"
    )
    log_path = tmp_path / document["methods"][2]["train"]["log_path"]
    log_path.write_text(
        "loading file:///home/example/private/checkpoint.pth\n", encoding="utf-8"
    )
    document["methods"][2]["train"]["log_sha256"] = hashlib.sha256(
        log_path.read_bytes()
    ).hexdigest()
    evidence = tmp_path / "private-paths.json"
    _write_document(evidence, document)

    result, report = _run(evidence)

    assert result.returncode != 0
    assert report["status"] == "not_validated"
    joined = " ".join(report["errors"]).lower()
    assert "raw path field" in joined
    assert "private absolute path" in joined


def test_runner_redacts_commands_and_logs_line_by_line(tmp_path):
    dataset_root = "/Users/example/private/mvtec_ad"
    work_dir = "/home/example/work/patchcore"
    replacements = {
        dataset_root: "<DATASET_ROOT>",
        work_dir: "<WORK_DIR>",
    }
    command = (
        f"python tools/train.py --dataset {dataset_root} --work-dir {work_dir} "
        r"--checkpoint C:\private\model.pth"
    )

    sanitized_command = _sanitize_text(command, replacements)

    assert dataset_root not in sanitized_command
    assert work_dir not in sanitized_command
    assert r"C:\private\model.pth" not in sanitized_command
    assert "<DATASET_ROOT>" in sanitized_command
    assert "<WORK_DIR>" in sanitized_command

    raw_log = tmp_path / "raw.log"
    public_log = tmp_path / "public.log"
    raw_log.write_text(
        f"dataset={dataset_root}\ncheckpoint=file:///home/example/private/model.pth\n",
        encoding="utf-8",
    )

    _sanitize_log(raw_log, public_log, replacements)

    sanitized_log = public_log.read_text(encoding="utf-8")
    assert not raw_log.exists()
    assert dataset_root not in sanitized_log
    assert "file:///home/example" not in sanitized_log
    assert "<DATASET_ROOT>" in sanitized_log
    assert "<ABSOLUTE_PATH>" in sanitized_log


def test_runner_cli_redacts_absolute_dataset_root_from_stderr_and_json(
    tmp_path, monkeypatch, capsys
):
    dataset_root = tmp_path / "private-dataset"
    output_dir = tmp_path / "evidence"

    def fake_git(*args):
        return "a" * 40 if args == ("rev-parse", "HEAD") else ""

    monkeypatch.setattr(gpu_runner, "_git", fake_git)

    result = gpu_runner.main(
        [
            "--dataset-root",
            str(dataset_root),
            "--output-dir",
            str(output_dir),
        ]
    )

    stderr = capsys.readouterr().err
    document = json.loads((output_dir / "gpu-evidence.json").read_text())
    assert result != 0
    assert str(dataset_root) not in stderr
    assert str(dataset_root) not in json.dumps(document)
    assert "<DATASET_ROOT>" in stderr
    assert "<DATASET_ROOT>" in document["reason"]


def test_runner_cli_redacts_failed_log_path_from_stderr_and_json(
    tmp_path, monkeypatch, capsys
):
    dataset_root = tmp_path / "mvtec_ad"
    (dataset_root / "bottle" / "train" / "good").mkdir(parents=True)
    output_dir = tmp_path / "private-work"
    private_log = output_dir / "logs" / "patchcore-train.log"

    def fake_git(*args):
        return "b" * 40 if args == ("rev-parse", "HEAD") else ""

    fake_cuda = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=fake_cuda))
    monkeypatch.setitem(sys.modules, "torchvision", types.SimpleNamespace())
    monkeypatch.setattr(gpu_runner, "_git", fake_git)
    monkeypatch.setattr(gpu_runner, "_compiled_cuda_ops", lambda *_: {})
    monkeypatch.setattr(gpu_runner, "_environment", lambda *_: {})
    monkeypatch.setattr(
        gpu_runner,
        "_run_phase",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError(f"failed; see {private_log}")
        ),
    )

    result = gpu_runner.main(
        [
            "--dataset-root",
            str(dataset_root),
            "--output-dir",
            str(output_dir),
        ]
    )

    stderr = capsys.readouterr().err
    document = json.loads((output_dir / "gpu-evidence.json").read_text())
    assert result != 0
    assert str(private_log) not in stderr
    assert str(private_log) not in json.dumps(document)
    assert "<EVIDENCE_DIR>" in stderr
    assert "<EVIDENCE_DIR>" in document["reason"]


def test_nvidia_driver_probe_uses_the_release_command(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="550.54.15\n", stderr="")

    monkeypatch.setattr(gpu_checker.subprocess, "run", fake_run)

    assert gpu_checker._nvidia_driver_version(0) == "550.54.15"
    assert calls[0][0] == [
        "nvidia-smi",
        "--query-gpu=driver_version",
        "--format=csv,noheader",
    ]


def test_live_driver_validation_rejects_missing_and_mismatched_driver(monkeypatch):
    environment = {"driver_version": "550.54.15"}
    errors = []
    monkeypatch.setattr(gpu_checker, "_nvidia_driver_version", lambda _index: None)
    gpu_checker._validate_live_driver(environment, 0, errors)
    assert errors == ["live NVIDIA driver version could not be determined"]

    errors = []
    monkeypatch.setattr(
        gpu_checker, "_nvidia_driver_version", lambda _index: "555.42.02"
    )
    gpu_checker._validate_live_driver(environment, 0, errors)
    assert errors == ["environment.driver_version does not match the live runtime"]

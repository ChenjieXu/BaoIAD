#!/usr/bin/env python3
"""Fail-closed validator for BaoIAD real-CUDA smoke evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
REQUIRED_METHODS = {"patchcore", "rd", "fastflow"}
REQUIRED_CUDA_CHECK = "torchvision_nms_cuda"
REQUIRED_STATE_CHECK = "mmcv_custom_ops"
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
CAPABILITY_PATTERN = re.compile(r"^[0-9]+\.[0-9]+$")
FILE_URI_PATTERN = re.compile(r"\bfile:(?://|\\\\)", re.IGNORECASE)
POSIX_ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?<![:A-Za-z0-9_])/(?!/)[A-Za-z0-9_.~+@%=-]+"
    r"(?:/[A-Za-z0-9_.~+@%=-]+)+"
)
WINDOWS_ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/][^\s'\"<>]+"
)
WINDOWS_UNC_PATH_PATTERN = re.compile(r"\\\\[A-Za-z0-9_.-]+\\[A-Za-z0-9_$.-]+")
RAW_PATH_FIELDS = {
    "checkpoint",
    "checkpoint_path",
    "data_root",
    "dataset_path",
    "dataset_root",
    "work_dir",
    "workdir",
}


def _git_commit(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _git_clean(root: Path) -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain=v2"],
        cwd=root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.returncode == 0 and not result.stdout.strip()


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def _contains_private_path(value: str) -> bool:
    return any(
        pattern.search(value)
        for pattern in (
            FILE_URI_PATTERN,
            POSIX_ABSOLUTE_PATH_PATTERN,
            WINDOWS_ABSOLUTE_PATH_PATTERN,
            WINDOWS_UNC_PATH_PATTERN,
        )
    )


def _validate_privacy_contract(
    value: Any, errors: list[str], location: str = "evidence"
) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized_key = str(key).lower().replace("-", "_")
            child_location = f"{location}.{key}"
            if normalized_key in RAW_PATH_FIELDS:
                errors.append(
                    f"raw path field is forbidden in GPU evidence: {child_location}"
                )
            _validate_privacy_contract(item, errors, child_location)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_privacy_contract(item, errors, f"{location}[{index}]")
    elif isinstance(value, str) and _contains_private_path(value):
        errors.append(f"private absolute path is forbidden in GPU evidence: {location}")


def _installed_mmcv_package() -> dict[str, str] | None:
    installed: list[tuple[str, str]] = []
    for package in ("mmcv", "mmcv-lite"):
        try:
            installed.append((package, metadata.version(package)))
        except metadata.PackageNotFoundError:
            continue
    if len(installed) != 1:
        return None
    package, version = installed[0]
    return {"package": package, "version": version}


def _nvidia_driver_version(device_index: int) -> str | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    versions = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if device_index < 0 or device_index >= len(versions):
        return None
    return versions[device_index]


def _validate_live_driver(
    environment: dict[str, Any], device_index: int, errors: list[str]
) -> None:
    live_driver = _nvidia_driver_version(device_index)
    if live_driver is None:
        errors.append("live NVIDIA driver version could not be determined")
    elif environment.get("driver_version") != live_driver:
        errors.append("environment.driver_version does not match the live runtime")


def _validate_phase(
    phase: Any, label: str, evidence_dir: Path, errors: list[str]
) -> int:
    if not isinstance(phase, dict):
        errors.append(f"{label} must be an object")
        return 0
    required = {
        "status",
        "command",
        "duration_seconds",
        "peak_vram_bytes",
        "log_path",
        "log_sha256",
    }
    if set(phase) != required:
        errors.append(f"{label} fields must be {sorted(required)}")
        return 0
    if phase.get("status") != "passed":
        errors.append(f"{label}.status must be passed")
    if not _non_empty_string(phase.get("command")):
        errors.append(f"{label}.command must be non-empty")
    if not _positive_number(phase.get("duration_seconds")):
        errors.append(f"{label}.duration_seconds must be positive")
    peak = phase.get("peak_vram_bytes")
    if not isinstance(peak, int) or isinstance(peak, bool) or peak <= 0:
        errors.append(f"{label}.peak_vram_bytes must be a positive integer")
        peak = 0

    log_path = phase.get("log_path")
    digest = phase.get("log_sha256")
    if not _non_empty_string(log_path):
        errors.append(f"{label}.log_path must be non-empty")
    else:
        relative = Path(log_path)
        if relative.is_absolute() or ".." in relative.parts:
            errors.append(f"{label}.log_path must stay inside the evidence directory")
        else:
            resolved = evidence_dir / relative
            if not resolved.is_file():
                errors.append(f"{label} log is missing: {log_path}")
            elif not isinstance(digest, str) or not DIGEST_PATTERN.fullmatch(digest):
                errors.append(f"{label}.log_sha256 must be a lowercase SHA-256")
            else:
                log_bytes = resolved.read_bytes()
                actual = hashlib.sha256(log_bytes).hexdigest()
                if actual != digest:
                    errors.append(
                        f"{label} log digest mismatch: expected {digest}, got {actual}"
                    )
                if _contains_private_path(log_bytes.decode("utf-8", errors="replace")):
                    errors.append(f"private absolute path is forbidden in {label} log")
    return peak


def _validate_document(
    document: Any, evidence_path: Path, root: Path
) -> tuple[list[str], dict[str, Any] | None]:
    errors: list[str] = []
    if not isinstance(document, dict):
        return ["evidence root must be an object"], None
    _validate_privacy_contract(document, errors)
    if document.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if document.get("status") not in {"validated", "not_validated"}:
        errors.append("status must be validated or not_validated")
    commit = document.get("commit_sha")
    if not isinstance(commit, str) or not SHA_PATTERN.fullmatch(commit):
        errors.append("commit_sha must be a lowercase 40-character Git SHA")
    generated_at = document.get("generated_at")
    if not _non_empty_string(generated_at):
        errors.append("generated_at must be an RFC 3339 timestamp")
    else:
        try:
            datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
        except ValueError:
            errors.append("generated_at must be an RFC 3339 timestamp")

    if document.get("status") == "not_validated":
        if not _non_empty_string(document.get("reason")):
            errors.append("not_validated evidence requires a reason")
        return errors, None

    required = {
        "schema_version",
        "status",
        "commit_sha",
        "generated_at",
        "runner",
        "repository_clean",
        "environment",
        "compiled_cuda_ops",
        "peak_vram_bytes",
        "methods",
    }
    if set(document) != required:
        errors.append(f"validated evidence fields must be {sorted(required)}")
    if not _non_empty_string(document.get("runner")):
        errors.append("runner must be non-empty")
    if document.get("repository_clean") is not True:
        errors.append("repository_clean must be true")
    peak = document.get("peak_vram_bytes")
    if not isinstance(peak, int) or isinstance(peak, bool) or peak <= 0:
        errors.append("peak_vram_bytes must be a positive integer")
        peak = 0

    environment = document.get("environment")
    if not isinstance(environment, dict):
        errors.append("environment must be an object")
        environment = {}
    required_environment_fields = {
        "cuda_available",
        "python_version",
        "torch_version",
        "torchvision_version",
        "cuda_runtime",
        "mmcv",
        "driver_version",
        "device",
    }
    allowed_environment_fields = required_environment_fields | {"driver_version"}
    for field in sorted(required_environment_fields - set(environment)):
        errors.append(f"environment.{field} is required")
    if set(environment) - allowed_environment_fields:
        errors.append(
            "environment fields are invalid: "
            f"{sorted(set(environment) - allowed_environment_fields)}"
        )
    if environment.get("cuda_available") is not True:
        errors.append("environment.cuda_available must be true")
    for key in (
        "python_version",
        "torch_version",
        "torchvision_version",
        "cuda_runtime",
        "driver_version",
    ):
        if not _non_empty_string(environment.get(key)):
            errors.append(f"environment.{key} must be non-empty")
    mmcv = environment.get("mmcv")
    if not isinstance(mmcv, dict):
        errors.append("environment.mmcv must be an object")
        mmcv = {}
    if set(mmcv) != {"package", "version"}:
        errors.append("environment.mmcv fields must be ['package', 'version']")
    if mmcv.get("package") not in {"mmcv", "mmcv-lite"}:
        errors.append("environment.mmcv.package must be mmcv or mmcv-lite")
    if not _non_empty_string(mmcv.get("version")):
        errors.append("environment.mmcv.version must be non-empty")
    device = environment.get("device")
    if not isinstance(device, dict):
        errors.append("environment.device must be an object")
        device = {}
    if not isinstance(device.get("index"), int) or isinstance(
        device.get("index"), bool
    ):
        errors.append("environment.device.index must be an integer")
    if not _non_empty_string(device.get("name")):
        errors.append("environment.device.name must be non-empty")
    capability = device.get("compute_capability")
    if not isinstance(capability, str) or not CAPABILITY_PATTERN.fullmatch(capability):
        errors.append("environment.device.compute_capability is invalid")
    if (
        not isinstance(device.get("total_memory_bytes"), int)
        or device.get("total_memory_bytes", 0) <= 0
    ):
        errors.append("environment.device.total_memory_bytes must be positive")

    compiled = document.get("compiled_cuda_ops")
    if not isinstance(compiled, dict):
        errors.append("compiled_cuda_ops must be an object")
        compiled = {}
    if compiled.get("status") not in {"available", "not_required"}:
        errors.append("compiled_cuda_ops.status must be available or not_required")
    checks = compiled.get("checks")
    check_by_name: dict[str, dict[str, Any]] = {}
    if not isinstance(checks, list):
        errors.append("compiled_cuda_ops.checks must be a list")
    else:
        for index, check in enumerate(checks):
            if not isinstance(check, dict):
                errors.append(f"compiled_cuda_ops.checks[{index}] must be an object")
                continue
            if set(check) != {"name", "required", "available", "detail"}:
                errors.append(f"compiled_cuda_ops.checks[{index}] fields are invalid")
                continue
            name = check.get("name")
            if not _non_empty_string(name) or name in check_by_name:
                errors.append(f"compiled_cuda_ops.checks[{index}].name is invalid")
                continue
            check_by_name[name] = check
            if not isinstance(check.get("required"), bool) or not isinstance(
                check.get("available"), bool
            ):
                errors.append(f"compiled_cuda_ops.checks[{index}] booleans are invalid")
            if check.get("required") is True and check.get("available") is not True:
                errors.append(f"required compiled CUDA op is unavailable: {name}")
            if not _non_empty_string(check.get("detail")):
                errors.append(f"compiled_cuda_ops.checks[{index}].detail is empty")
    required_check = check_by_name.get(REQUIRED_CUDA_CHECK)
    if not required_check or required_check.get("required") is not True:
        errors.append(f"missing required CUDA op check: {REQUIRED_CUDA_CHECK}")
    if REQUIRED_STATE_CHECK not in check_by_name:
        errors.append(f"missing compiled-op state check: {REQUIRED_STATE_CHECK}")

    methods = document.get("methods")
    names: set[str] = set()
    phase_peaks: list[int] = []
    if not isinstance(methods, list):
        errors.append("methods must be a list")
    else:
        for index, method in enumerate(methods):
            if not isinstance(method, dict):
                errors.append(f"methods[{index}] must be an object")
                continue
            required_method_fields = {
                "name",
                "config",
                "dataset",
                "category",
                "train",
                "inference",
            }
            if set(method) != required_method_fields:
                errors.append(f"methods[{index}] fields are invalid")
                continue
            name = method.get("name")
            if name in names:
                errors.append(f"duplicate GPU smoke method: {name}")
            if isinstance(name, str):
                names.add(name)
            for key in ("config", "dataset", "category"):
                if not _non_empty_string(method.get(key)):
                    errors.append(f"methods[{index}].{key} must be non-empty")
            config = method.get("config")
            if isinstance(config, str) and not (root / config).is_file():
                errors.append(f"GPU smoke config is missing: {config}")
            phase_peaks.append(
                _validate_phase(
                    method.get("train"),
                    f"methods[{index}].train",
                    evidence_path.parent,
                    errors,
                )
            )
            phase_peaks.append(
                _validate_phase(
                    method.get("inference"),
                    f"methods[{index}].inference",
                    evidence_path.parent,
                    errors,
                )
            )
    if names != REQUIRED_METHODS:
        errors.append(f"GPU smoke methods must be exactly {sorted(REQUIRED_METHODS)}")
    if phase_peaks and max(phase_peaks) != peak:
        errors.append("peak_vram_bytes must equal the maximum train/inference peak")
    return errors, {"environment": environment, "device": device}


def _validate_live_cuda(
    document: dict[str, Any], root: Path, errors: list[str]
) -> None:
    try:
        import torch
        import torchvision
    except Exception as exc:  # pragma: no cover - environment dependent
        errors.append(f"PyTorch/TorchVision CUDA probe could not import runtime: {exc}")
        return
    if not torch.cuda.is_available():
        errors.append(
            "real CUDA is not available; CPU execution cannot validate GPU evidence"
        )
        return

    environment = document["environment"]
    device = environment["device"]
    index = device["index"]
    if index >= torch.cuda.device_count():
        errors.append(f"recorded CUDA device index does not exist: {index}")
        return
    properties = torch.cuda.get_device_properties(index)
    live_capability = f"{properties.major}.{properties.minor}"
    live_checks = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    for key, actual in live_checks.items():
        if environment.get(key) != actual:
            errors.append(
                f"environment.{key} does not match live runtime: "
                f"evidence={environment.get(key)!r}, live={actual!r}"
            )
    live_mmcv = _installed_mmcv_package()
    if live_mmcv is None:
        errors.append("live runtime must contain exactly one of mmcv or mmcv-lite")
    elif environment.get("mmcv") != live_mmcv:
        errors.append("environment.mmcv does not match the live runtime")
    _validate_live_driver(environment, index, errors)
    if device.get("name") != properties.name:
        errors.append("recorded CUDA device name does not match live runtime")
    if device.get("compute_capability") != live_capability:
        errors.append("recorded CUDA compute capability does not match live runtime")
    if device.get("total_memory_bytes") != properties.total_memory:
        errors.append("recorded CUDA total memory does not match live runtime")

    try:
        boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]], device=f"cuda:{index}")
        scores = torch.tensor([1.0], device=f"cuda:{index}")
        torchvision.ops.nms(boxes, scores, 0.5)
        torch.cuda.synchronize(index)
    except Exception as exc:  # pragma: no cover - requires CUDA
        errors.append(f"live torchvision CUDA NMS probe failed: {exc}")

    commit = _git_commit(root)
    if commit != document.get("commit_sha"):
        errors.append(
            f"evidence commit does not match checked-out commit: "
            f"evidence={document.get('commit_sha')!r}, live={commit!r}"
        )
    if not _git_clean(root) or document.get("repository_clean") is not True:
        errors.append("GPU evidence must be produced from a clean exact commit")


def validate_gpu_evidence(
    evidence_path: Path | None, root: Path = ROOT
) -> dict[str, Any]:
    if evidence_path is None or not evidence_path.is_file():
        return {
            "ok": False,
            "status": "not_validated",
            "errors": ["GPU evidence file is missing; GPU support is not validated"],
        }
    try:
        document = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "ok": False,
            "status": "not_validated",
            "errors": [f"GPU evidence cannot be read: {exc}"],
        }
    errors, _ = _validate_document(document, evidence_path, root)
    if (
        isinstance(document, dict)
        and document.get("status") == "validated"
        and not errors
    ):
        _validate_live_cuda(document, root, errors)
    status = (
        "validated"
        if not errors and document.get("status") == "validated"
        else "not_validated"
    )
    return {"ok": status == "validated", "status": status, "errors": errors}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument(
        "--require",
        "--require-validated",
        dest="require_validated",
        action="store_true",
        help="Return non-zero when real-GPU evidence is not validated.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = validate_gpu_evidence(args.evidence, args.repo_root.resolve())
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    elif report["ok"]:
        print("GPU smoke evidence: PASS (real CUDA validated)")
    else:
        print("GPU smoke evidence: NOT VALIDATED", file=sys.stderr)
        for error in report["errors"]:
            print(f"- {error}", file=sys.stderr)
    if report["ok"]:
        return 0
    return 1 if args.require_validated or report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the release GPU smoke matrix and emit auditable evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CATEGORY = "bottle"
METHODS = {
    "patchcore": "configs/patchcore/patchcore_wrn50_256_mvtec_strict.py",
    "rd": "configs/rd/rd_wrn50_256_mvtec_strict.py",
    "fastflow": "configs/fastflow/fastflow_wrn50_256_mvtec_strict.py",
}
FILE_URI_PATTERN = re.compile(r"\bfile:(?://|\\\\)[^\s'\"<>]+", re.IGNORECASE)
POSIX_ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?<![:A-Za-z0-9_])/(?!/)[A-Za-z0-9_.~+@%=-]+"
    r"(?:/[A-Za-z0-9_.~+@%=-]+)+"
)
WINDOWS_ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/][^\s'\"<>]+"
)
WINDOWS_UNC_PATH_PATTERN = re.compile(
    r"\\\\[A-Za-z0-9_.-]+\\[A-Za-z0-9_$.-]+(?:\\[^\s'\"<>]+)*"
)


class GPUValidationError(RuntimeError):
    """Sanitized GPU validation failure safe for evidence and stderr."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _not_validated(path: Path, commit: str, reason: str) -> None:
    _write_json(
        path,
        {
            "schema_version": 1,
            "status": "not_validated",
            "reason": reason,
            "commit_sha": commit,
            "generated_at": _now(),
        },
    )


def _sanitize_text(value: str, replacements: dict[str, str]) -> str:
    sanitized = value
    for source, replacement in sorted(
        replacements.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if source:
            sanitized = sanitized.replace(source, replacement)
    for pattern in (
        FILE_URI_PATTERN,
        WINDOWS_UNC_PATH_PATTERN,
        WINDOWS_ABSOLUTE_PATH_PATTERN,
        POSIX_ABSOLUTE_PATH_PATTERN,
    ):
        sanitized = pattern.sub("<ABSOLUTE_PATH>", sanitized)
    return sanitized


def _base_replacements(dataset_root: Path, output_dir: Path) -> dict[str, str]:
    return {
        str(dataset_root): "<DATASET_ROOT>",
        str(output_dir): "<EVIDENCE_DIR>",
        str(ROOT): ".",
        sys.executable: "python",
    }


def _sanitize_log(
    raw_log: Path,
    public_log: Path,
    replacements: dict[str, str],
) -> None:
    public_log.parent.mkdir(parents=True, exist_ok=True)
    try:
        with raw_log.open("r", encoding="utf-8", errors="replace") as source:
            with public_log.open("w", encoding="utf-8") as destination:
                for line in source:
                    destination.write(_sanitize_text(line, replacements))
    finally:
        raw_log.unlink(missing_ok=True)


def _installed_mmcv_package() -> dict[str, str]:
    installed: list[tuple[str, str]] = []
    for package in ("mmcv", "mmcv-lite"):
        try:
            installed.append((package, metadata.version(package)))
        except metadata.PackageNotFoundError:
            continue
    if len(installed) != 1:
        raise RuntimeError("exactly one of mmcv or mmcv-lite must be installed")
    package, version = installed[0]
    return {"package": package, "version": version}


def _nvidia_smi(*query: str) -> str | None:
    result = subprocess.run(
        ["nvidia-smi", *query],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _process_vram_bytes(pid: int) -> int:
    output = _nvidia_smi(
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    )
    if not output:
        return 0
    peak_mib = 0
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            row_pid, used_mib = (int(field) for field in fields)
        except ValueError:
            continue
        if row_pid == pid:
            peak_mib = max(peak_mib, used_mib)
    return peak_mib * 1024 * 1024


def _run_phase(
    command: list[str],
    log_path: Path,
    env: dict[str, str],
    replacements: dict[str, str],
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    peak_vram = 0
    display_command = _sanitize_text(shlex.join(command), replacements)
    phase_error: Exception | None = None
    returncode: int | None = None
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="baoiad-gpu-",
        suffix=".raw.log",
        delete=False,
    ) as stream:
        raw_log = Path(stream.name)
        stream.write(f"$ {display_command}\n")
        stream.flush()
        try:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                text=True,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
            while process.poll() is None:
                peak_vram = max(peak_vram, _process_vram_bytes(process.pid))
                time.sleep(0.2)
            returncode = process.returncode
        except Exception as exc:
            phase_error = exc
        finally:
            stream.flush()
    _sanitize_log(raw_log, log_path, replacements)
    if phase_error is not None:
        raise phase_error
    duration = time.monotonic() - started
    if returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {returncode}; see {log_path}"
        )
    if peak_vram <= 0:
        raise RuntimeError(f"no process VRAM was observed; see {log_path}")
    return {
        "status": "passed",
        "command": display_command,
        "duration_seconds": round(duration, 3),
        "peak_vram_bytes": peak_vram,
        "log_path": log_path.as_posix(),
        "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
    }


def _dataset_overrides(dataset_root: Path) -> list[str]:
    overrides: list[str] = []
    for loader in ("train_dataloader", "val_dataloader", "test_dataloader"):
        prefix = f"{loader}.dataset"
        overrides.extend(
            [
                f"{prefix}.data_root={str(dataset_root)!r}",
                f"{prefix}.cls_names=['{CATEGORY}']",
                f"{prefix}.multi_class=False",
                f"{loader}.batch_size=1",
                f"{loader}.num_workers=0",
                f"{loader}.persistent_workers=False",
            ]
        )
    return overrides


def _train_command(config: str, work_dir: Path, dataset_root: Path) -> list[str]:
    return [
        sys.executable,
        "tools/train.py",
        config,
        "--work-dir",
        str(work_dir),
        "--offline",
        "--cfg-options",
        *_dataset_overrides(dataset_root),
        "train_cfg.max_epochs=1",
        "train_cfg.val_begin=1",
        "train_cfg.val_interval=1",
        "default_hooks.checkpoint.interval=1",
        "default_hooks.checkpoint.max_keep_ckpts=1",
    ]


def _inference_command(
    config: str, checkpoint: Path, work_dir: Path, dataset_root: Path
) -> list[str]:
    return [
        sys.executable,
        "tools/test.py",
        config,
        str(checkpoint),
        "--work-dir",
        str(work_dir),
        "--offline",
        "--trusted-checkpoint",
        "--cfg-options",
        *_dataset_overrides(dataset_root),
    ]


def _compiled_cuda_ops(torch: Any, device_index: int) -> dict[str, Any]:
    import torchvision

    boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]], device=f"cuda:{device_index}")
    scores = torch.tensor([1.0], device=f"cuda:{device_index}")
    torchvision.ops.nms(boxes, scores, 0.5)
    torch.cuda.synchronize(device_index)

    mmcv_available = False
    mmcv_detail = "mmcv custom ops are not installed and are not required"
    try:
        from mmcv import ops as mmcv_ops  # noqa: F401

        mmcv_available = True
        mmcv_detail = (
            "mmcv custom ops imported successfully; not required by smoke methods"
        )
    except (ImportError, ModuleNotFoundError) as exc:
        mmcv_detail = (
            f"mmcv custom ops unavailable and not required: {type(exc).__name__}"
        )

    return {
        "status": "available",
        "checks": [
            {
                "name": "torchvision_nms_cuda",
                "required": True,
                "available": True,
                "detail": "torchvision.ops.nms executed on the recorded CUDA device",
            },
            {
                "name": "mmcv_custom_ops",
                "required": False,
                "available": mmcv_available,
                "detail": mmcv_detail,
            },
        ],
    }


def _environment(torch: Any, torchvision: Any, device_index: int) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device_index)
    environment: dict[str, Any] = {
        "cuda_available": True,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "mmcv": _installed_mmcv_package(),
        "device": {
            "index": device_index,
            "name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_bytes": properties.total_memory,
        },
    }
    driver = _nvidia_smi(
        f"--id={device_index}",
        "--query-gpu=driver_version",
        "--format=csv,noheader",
    )
    if not driver:
        raise RuntimeError("NVIDIA driver version could not be determined")
    environment["driver_version"] = driver.splitlines()[0].strip()
    return environment


def _checkpoint(work_dir: Path) -> Path:
    checkpoints = sorted(work_dir.glob("*.pth"), key=lambda path: path.stat().st_mtime)
    if not checkpoints:
        raise RuntimeError(f"training did not produce a checkpoint in {work_dir}")
    return checkpoints[-1]


def run(dataset_root: Path, output_dir: Path, device_index: int) -> Path:
    evidence_path = output_dir / "gpu-evidence.json"
    base_replacements = _base_replacements(dataset_root, output_dir)
    try:
        commit = _git("rev-parse", "HEAD")
    except (OSError, subprocess.SubprocessError) as exc:
        message = _sanitize_text(
            f"cannot determine the checked-out commit: {exc}", base_replacements
        )
        raise GPUValidationError(message) from None

    try:
        if _git("status", "--porcelain=v2"):
            raise RuntimeError("the repository is not clean")
        if not (dataset_root / CATEGORY / "train" / "good").is_dir():
            raise RuntimeError(
                f"MVTec AD {CATEGORY} data is missing under {dataset_root}"
            )

        import torch
        import torchvision

        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is false")
        if device_index < 0 or device_index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device index {device_index} does not exist")
        compiled_cuda_ops = _compiled_cuda_ops(torch, device_index)
        environment = _environment(torch, torchvision, device_index)

        child_env = os.environ.copy()
        child_env["BAOIAD_OFFLINE"] = "1"
        methods: list[dict[str, Any]] = []
        phase_peaks: list[int] = []
        for name, config in METHODS.items():
            method_dir = output_dir / "runs" / name
            train_work_dir = method_dir / "train"
            inference_work_dir = method_dir / "inference"
            train_log = output_dir / "logs" / f"{name}-train.log"
            inference_log = output_dir / "logs" / f"{name}-inference.log"
            train = _run_phase(
                _train_command(config, train_work_dir, dataset_root),
                train_log,
                child_env,
                {
                    **base_replacements,
                    str(train_work_dir): "<WORK_DIR>",
                },
            )
            checkpoint = _checkpoint(train_work_dir)
            inference = _run_phase(
                _inference_command(
                    config, checkpoint, inference_work_dir, dataset_root
                ),
                inference_log,
                child_env,
                {
                    **base_replacements,
                    str(checkpoint): "<CHECKPOINT>",
                    str(inference_work_dir): "<WORK_DIR>",
                },
            )
            for phase in (train, inference):
                phase["log_path"] = str(Path(phase["log_path"]).relative_to(output_dir))
                phase_peaks.append(phase["peak_vram_bytes"])
            methods.append(
                {
                    "name": name,
                    "config": config,
                    "dataset": "MVTec AD",
                    "category": CATEGORY,
                    "train": train,
                    "inference": inference,
                }
            )

        if _git("status", "--porcelain=v2"):
            raise RuntimeError("the smoke run changed the checked-out repository")
        _write_json(
            evidence_path,
            {
                "schema_version": 1,
                "status": "validated",
                "commit_sha": commit,
                "generated_at": _now(),
                "runner": os.environ.get("RUNNER_NAME", "local-gpu-runner"),
                "repository_clean": True,
                "environment": environment,
                "compiled_cuda_ops": compiled_cuda_ops,
                "peak_vram_bytes": max(phase_peaks),
                "methods": methods,
            },
        )
    except Exception as exc:
        message = _sanitize_text(str(exc), base_replacements)
        _not_validated(
            evidence_path,
            commit,
            message,
        )
        raise GPUValidationError(message) from None
    return evidence_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device-index", type=int, default=0)
    args = parser.parse_args(argv)
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    replacements = _base_replacements(dataset_root, output_dir)
    try:
        evidence = run(dataset_root, output_dir, args.device_index)
    except Exception as exc:
        message = _sanitize_text(str(exc), replacements)
        print(f"GPU smoke did not validate: {message}", file=sys.stderr)
        return 1
    print(evidence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Lightweight, network-free BaoIAD installation verification."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import platform
import sys
from collections.abc import Callable, Mapping, MutableMapping
from importlib import metadata
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CORE_PACKAGES = (
    ("BaoIAD", "baoiad", ("baoiad",)),
    ("PyTorch", "torch", ("torch",)),
    ("MMEngine", "mmengine", ("mmengine",)),
    ("MMCV", "mmcv", ("mmcv", "mmcv-lite")),
)

# Keep these import names synchronized with ``project.optional-dependencies``
# in pyproject.toml. Availability is intentionally checked without importing
# the optional packages.
OPTIONAL_GROUP_MODULES = {
    "dev": ("pytest", "ruff", "pre_commit"),
    "flow": ("FrEIA", "networkx", "mpmath", "skimage"),
    "vl": ("open_clip",),
    "saa": ("groundingdino", "segment_anything"),
    "geomloss": ("geomloss",),
    "glass": ("pandas", "openpyxl"),
    "visualization": ("matplotlib",),
    "faiss-cpu": ("faiss",),
}
ALL_EXTRA_GROUPS = (
    "flow",
    "vl",
    "saa",
    "geomloss",
    "glass",
    "visualization",
    "faiss-cpu",
)
OFFLINE_ENV_VARS = (
    "BAOIAD_OFFLINE",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_DATASETS_OFFLINE",
)
TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def _distribution_version(candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        try:
            return metadata.version(candidate)
        except metadata.PackageNotFoundError:
            continue
    return None


def inspect_core_packages(
    *, import_module: Callable[[str], Any] = importlib.import_module
) -> list[dict[str, Any]]:
    """Import only the four core packages and report their versions."""
    checks: list[dict[str, Any]] = []
    for label, module_name, distributions in CORE_PACKAGES:
        try:
            module = import_module(module_name)
        except Exception as exc:  # pragma: no cover - exact dependency errors vary
            checks.append(
                {
                    "name": label,
                    "module": module_name,
                    "available": False,
                    "version": _distribution_version(distributions),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        version = getattr(module, "__version__", None)
        if version is None:
            version = _distribution_version(distributions)
        checks.append(
            {
                "name": label,
                "module": module_name,
                "available": True,
                "version": str(version) if version is not None else "unknown",
                "error": None,
            }
        )
    return checks


def _module_available(
    module_name: str,
    *,
    find_spec: Callable[[str], Any] = importlib.util.find_spec,
) -> bool:
    try:
        return find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def inspect_optional_groups(
    *, find_spec: Callable[[str], Any] = importlib.util.find_spec
) -> list[dict[str, Any]]:
    """Report optional extras without importing or initializing them."""
    group_modules = dict(OPTIONAL_GROUP_MODULES)
    group_modules["all"] = tuple(
        dict.fromkeys(
            module
            for group in ALL_EXTRA_GROUPS
            for module in OPTIONAL_GROUP_MODULES[group]
        )
    )

    checks: list[dict[str, Any]] = []
    for group, modules in group_modules.items():
        missing = [
            module
            for module in modules
            if not _module_available(module, find_spec=find_spec)
        ]
        checks.append(
            {
                "group": group,
                "available": not missing,
                "missing_modules": missing,
            }
        )
    return checks


def _absolute_path(value: str | os.PathLike[str], *, base: Path) -> str:
    raw = os.fspath(value)
    if "\x00" in raw:
        raise ValueError("path must not contain NUL bytes")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base / path
    return str(path.resolve(strict=False))


def resolve_paths(
    environ: Mapping[str, str],
    *,
    import_module: Callable[[str], Any] = importlib.import_module,
) -> tuple[dict[str, str | None], list[str]]:
    """Resolve data and cache paths without inspecting their contents."""
    errors: list[str] = []
    try:
        paths_module = import_module("baoiad.paths")
        data_root = str(
            paths_module.get_data_root(
                environ=environ,
                repository_root=ROOT,
            )
        )
    except Exception as exc:
        data_root = None
        errors.append(f"data root: {type(exc).__name__}: {exc}")

    home_raw = environ.get("HOME") or str(Path.home())
    try:
        home = Path(_absolute_path(home_raw, base=ROOT))
        xdg_raw = environ.get("XDG_CACHE_HOME")
        cache_base = (
            Path(_absolute_path(xdg_raw, base=Path.cwd()))
            if xdg_raw and xdg_raw.strip()
            else home / ".cache"
        )
        hf_raw = environ.get("HF_HOME")
        torch_raw = environ.get("TORCH_HOME")
        baoiad_raw = environ.get("BAOIAD_CACHE_DIR")
        hf_home = _absolute_path(
            hf_raw if hf_raw and hf_raw.strip() else cache_base / "huggingface",
            base=Path.cwd(),
        )
        torch_home = _absolute_path(
            torch_raw if torch_raw and torch_raw.strip() else cache_base / "torch",
            base=Path.cwd(),
        )
        baoiad_cache = (
            _absolute_path(baoiad_raw, base=Path.cwd())
            if baoiad_raw and baoiad_raw.strip()
            else None
        )
    except (OSError, TypeError, ValueError) as exc:
        hf_home = None
        torch_home = None
        baoiad_cache = None
        errors.append(f"cache paths: {type(exc).__name__}: {exc}")

    return (
        {
            "data_root": data_root,
            "hf_home": hf_home,
            "torch_home": torch_home,
            "baoiad_cache_dir": baoiad_cache,
        },
        errors,
    )


def _offline_enabled(environ: Mapping[str, str]) -> bool:
    return any(
        str(environ.get(name, "")).strip().lower() in TRUE_VALUES
        for name in OFFLINE_ENV_VARS
    )


def _enable_offline(environ: MutableMapping[str, str]) -> None:
    for name in OFFLINE_ENV_VARS:
        environ[name] = "1"


def collect_report(
    environ: Mapping[str, str],
    *,
    import_module: Callable[[str], Any] = importlib.import_module,
    find_spec: Callable[[str], Any] = importlib.util.find_spec,
) -> dict[str, Any]:
    core = inspect_core_packages(import_module=import_module)
    paths, path_errors = resolve_paths(environ, import_module=import_module)
    optional = inspect_optional_groups(find_spec=find_spec)
    ok = all(item["available"] for item in core) and not path_errors
    return {
        "ok": ok,
        "core": [
            {
                "name": "Python",
                "module": "python",
                "available": True,
                "version": platform.python_version(),
                "error": None,
            },
            *core,
        ],
        "paths": paths,
        "optional_groups": optional,
        "runtime_policy": {
            "offline": _offline_enabled(environ),
            "network_access": "not attempted",
            "dataset_access": "not attempted",
            "checkpoint_loading": "not attempted",
            "gpu_probe": "not attempted",
        },
        "errors": path_errors,
    }


def _print_text(report: dict[str, Any]) -> None:
    print("BaoIAD installation verification")
    print("\nCore versions")
    for item in report["core"]:
        status = "available" if item["available"] else "unavailable"
        version = item["version"] or "unknown"
        print(f"- {item['name']}: {version} ({status})")
        if item["error"]:
            print(f"  error: {item['error']}")

    print("\nResolved paths")
    for name, value in report["paths"].items():
        print(f"- {name}: {value if value is not None else 'not configured'}")

    print("\nOptional dependency groups")
    for item in report["optional_groups"]:
        if item["available"]:
            detail = "available"
        else:
            detail = "unavailable; missing " + ", ".join(item["missing_modules"])
        print(f"- {item['group']}: {detail}")

    policy = report["runtime_policy"]
    print("\nRuntime policy")
    print(f"- offline: {'enabled' if policy['offline'] else 'disabled'}")
    print("- network, datasets, checkpoints, and GPU: not accessed")
    if report["errors"]:
        print("\nErrors")
        for error in report["errors"]:
            print(f"- {error}")
    print("\nPASS" if report["ok"] else "\nFAIL")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a BaoIAD installation without data, downloads, or GPU work."
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Enable BaoIAD and supported model-hub offline variables for this process.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the verification report as JSON.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.offline:
        _enable_offline(os.environ)
    report = collect_report(os.environ)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        _print_text(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

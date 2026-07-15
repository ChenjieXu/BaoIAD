"""Import-safe filesystem path resolution for BaoIAD."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

DATA_ROOT_ENV = "BAOIAD_DATA_ROOT"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPOSITORY_ROOT / "data"


def _normalize_path(value: str | os.PathLike[str], *, repository_root: Path) -> Path:
    raw_value = os.fspath(value)
    if not isinstance(raw_value, str):
        raise TypeError("data root must be a text path")

    normalized_value = raw_value.strip()
    if not normalized_value:
        raise ValueError("data root must not be empty")
    if "\x00" in normalized_value:
        raise ValueError("data root must not contain NUL bytes")

    path = Path(normalized_value).expanduser()
    if not path.is_absolute():
        path = repository_root / path
    return path.resolve(strict=False)


def get_data_root(
    *,
    environ: Mapping[str, str] | None = None,
    repository_root: str | os.PathLike[str] | None = None,
) -> Path:
    """Return the top-level dataset directory.

    ``BAOIAD_DATA_ROOT`` overrides the repository-local ``data/`` default.
    Relative values are resolved against the repository root, never the
    process's current working directory.
    """
    environment = os.environ if environ is None else environ
    repo_root = (
        REPOSITORY_ROOT
        if repository_root is None
        else Path(repository_root).expanduser().resolve(strict=False)
    )
    configured_root = environment.get(DATA_ROOT_ENV)
    if configured_root is None:
        return repo_root / "data"
    return _normalize_path(configured_root, repository_root=repo_root)


def resolve_data_root(
    dataset_name: str,
    *,
    override: str | os.PathLike[str] | None = None,
    environ: Mapping[str, str] | None = None,
    repository_root: str | os.PathLike[str] | None = None,
) -> Path:
    """Resolve one dataset path with CLI-over-environment precedence.

    ``override`` represents the final dataset path supplied by a CLI config
    override. Without it, ``dataset_name`` is appended to the top-level data
    root returned by :func:`get_data_root`.
    """
    repo_root = (
        REPOSITORY_ROOT
        if repository_root is None
        else Path(repository_root).expanduser().resolve(strict=False)
    )
    if override is not None:
        return _normalize_path(override, repository_root=repo_root)

    normalized_name = dataset_name.strip()
    dataset_path = Path(normalized_name)
    if (
        not normalized_name
        or "\x00" in normalized_name
        or normalized_name in {".", ".."}
        or dataset_path.is_absolute()
        or ".." in dataset_path.parts
    ):
        raise ValueError(f"invalid dataset name: {dataset_name!r}")
    return get_data_root(environ=environ, repository_root=repo_root) / dataset_path


__all__ = [
    "DATA_ROOT_ENV",
    "DEFAULT_DATA_ROOT",
    "REPOSITORY_ROOT",
    "get_data_root",
    "resolve_data_root",
]

"""Import-safe runtime configuration normalization."""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping, MutableSequence
from pathlib import Path
from typing import Any

from baoiad.paths import get_data_root


def _resolved_config_data_root(value: Any, *, top_level_root: Path) -> Any:
    if not isinstance(value, (str, os.PathLike)):
        return value

    path = Path(os.fspath(value)).expanduser()
    if path.is_absolute():
        return str(path.resolve(strict=False))
    if not path.parts or path.parts[0] != "data":
        return value
    return str((top_level_root.joinpath(*path.parts[1:])).resolve(strict=False))


def apply_data_root_overrides(
    config: Any,
    *,
    environ: Mapping[str, str] | None = None,
    repository_root: str | os.PathLike[str] | None = None,
) -> None:
    """Resolve repository-relative roots in a dict or MMEngine ``Config``.

    Call this immediately after loading a config and before applying explicit
    CLI ``--cfg-options`` so command-line paths keep the highest precedence.
    """
    top_level_root = get_data_root(environ=environ, repository_root=repository_root)
    target = getattr(config, "_cfg_dict", config)
    if not isinstance(target, MutableMapping):
        raise TypeError(
            "config must be a mutable mapping or expose MMEngine _cfg_dict"
        )

    def visit(value: Any) -> None:
        if isinstance(value, MutableMapping):
            for key, item in list(value.items()):
                if key in {"data_root", "support_set_root"}:
                    value[key] = _resolved_config_data_root(
                        item, top_level_root=top_level_root
                    )
                else:
                    visit(item)
        elif isinstance(value, MutableSequence):
            for item in value:
                visit(item)

    visit(target)


__all__ = ["apply_data_root_overrides"]

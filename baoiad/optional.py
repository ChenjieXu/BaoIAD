"""Helpers for explicit, actionable optional-dependency failures."""

from __future__ import annotations

import importlib
from types import ModuleType


class OptionalDependencyError(ModuleNotFoundError):
    """Raised when a selected BaoIAD feature needs an uninstalled extra."""


def optional_dependency_message(*, extra: str, feature: str, import_name: str) -> str:
    """Build the canonical installation guidance for one optional feature."""
    return (
        f"{feature} requires the optional dependency {import_name!r}. "
        f'Install it with: pip install -e ".[{extra}]"'
    )


def require_optional_module(
    import_name: str, *, extra: str, feature: str
) -> ModuleType:
    """Import an optional module or raise an error naming its public extra."""
    try:
        return importlib.import_module(import_name)
    except ModuleNotFoundError as exc:
        raise OptionalDependencyError(
            optional_dependency_message(
                extra=extra, feature=feature, import_name=import_name
            ),
            name=exc.name or import_name,
        ) from exc


__all__ = [
    "OptionalDependencyError",
    "optional_dependency_message",
    "require_optional_module",
]

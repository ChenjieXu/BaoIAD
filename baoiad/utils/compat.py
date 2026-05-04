"""Compatibility helpers for legacy third-party dependencies."""

from __future__ import annotations

import collections
import collections.abc

import numpy as np


def ensure_legacy_imgaug_compat() -> None:
    """Provide NumPy / collections shims required by legacy imgaug code paths."""
    if not hasattr(collections, 'Iterable'):
        collections.Iterable = collections.abc.Iterable  # type: ignore[attr-defined]
    if not hasattr(collections, 'Mapping'):
        collections.Mapping = collections.abc.Mapping  # type: ignore[attr-defined]
    if not hasattr(collections, 'MutableMapping'):
        collections.MutableMapping = collections.abc.MutableMapping  # type: ignore[attr-defined]
    if not hasattr(collections, 'Sequence'):
        collections.Sequence = collections.abc.Sequence  # type: ignore[attr-defined]

    if not hasattr(np, 'sctypes'):
        np.sctypes = {  # type: ignore[attr-defined]
            'int': [np.int8, np.int16, np.int32, np.int64],
            'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
            'float': [np.float16, np.float32, np.float64],
            'complex': [np.complex64, np.complex128],
            'others': [np.bool_, np.object_, np.str_, np.bytes_],
        }


def patch_numpy_sctypes() -> None:
    """Backward-compatible alias for callers that only expect the NumPy shim."""
    ensure_legacy_imgaug_compat()

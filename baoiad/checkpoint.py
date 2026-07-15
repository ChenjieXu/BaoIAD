"""Checkpoint loading policy with safe defaults and explicit legacy opt-in."""

from __future__ import annotations

import os
import pickle
import re
import threading
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any


_TRUSTED_CHECKPOINTS = ContextVar('baoiad_trusted_checkpoints', default=False)
_TORCH_FORCE_SAFE = 'TORCH_FORCE_WEIGHTS_ONLY_LOAD'
_TORCH_FORCE_UNSAFE = 'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'
_POLICY_LOCK = threading.RLock()
_MISSING = object()


class CheckpointError(RuntimeError):
    """Base class for checkpoint policy and loading failures."""


class UnsafeCheckpointError(CheckpointError):
    """Raised when a checkpoint cannot be loaded with the safe policy."""


class CheckpointLoadError(CheckpointError):
    """Raised when a checkpoint is corrupt or operationally unreadable."""


class TrustedCheckpointWarning(UserWarning):
    """Warn that a trusted legacy checkpoint may execute Python code."""


def trusted_checkpoint_loading_enabled() -> bool:
    """Return whether the current policy explicitly trusts legacy pickle data."""
    return bool(_TRUSTED_CHECKPOINTS.get())


def _restore_environment(name: str, previous: str | None) -> None:
    if previous is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous


def _history_buffer_getattr(obj: Any, name: str) -> Any:
    """Resolve only methods serialized by MMEngine ``HistoryBuffer``."""
    from mmengine.logging import HistoryBuffer

    allowed_names = {'current', 'max', 'mean', 'min'}
    if obj is not HistoryBuffer or name not in allowed_names:
        raise UnsafeCheckpointError(
            f'Blocked checkpoint attribute lookup: {obj!r}.{name}'
        )
    return getattr(obj, name)


def _is_restricted_unpickling_error(exc: pickle.UnpicklingError) -> bool:
    """Distinguish a blocked Python object from malformed serialization."""
    message = str(exc)
    return any(
        marker in message
        for marker in (
            'Unsupported global:',
            'types allowlisted via `add_safe_globals`',
            'Trying to call reduce for unrecognized function',
            'Blocked checkpoint attribute lookup',
        )
    )


def _supports_named_safe_globals(torch_module: Any) -> bool:
    """Return whether PyTorch supports ``(callable, serialized_name)`` aliases."""
    match = re.match(r'^(\d+)\.(\d+)', str(getattr(torch_module, '__version__', '')))
    if match is None:
        return False
    return tuple(map(int, match.groups())) >= (2, 6)


@contextmanager
def _restricted_safe_globals(torch_module: Any) -> Iterator[None]:
    """Allow the narrow metadata types in BaoIAD MMEngine checkpoints."""
    safe_globals = getattr(torch_module.serialization, 'safe_globals', None)
    if safe_globals is None or not _supports_named_safe_globals(torch_module):
        yield
        return

    import numpy as np
    from mmengine.logging import HistoryBuffer

    try:
        from numpy._core.multiarray import _reconstruct
    except ImportError:  # NumPy < 2
        from numpy.core.multiarray import _reconstruct

    allowed = [
        HistoryBuffer,
        np.ndarray,
        np.dtype,
        type(np.dtype(np.float64)),
        type(np.dtype(np.int64)),
        _reconstruct,
        (_history_buffer_getattr, 'builtins.getattr'),
    ]
    with safe_globals(allowed):
        yield


@contextmanager
def checkpoint_loading_policy(trusted: bool = False) -> Iterator[None]:
    """Apply one scoped policy to PyTorch, MMEngine, and BaoIAD loaders.

    The default forces ``weights_only=True``. ``trusted=True`` is intended only
    for checkpoints whose origin and integrity were independently verified.
    """
    trusted = bool(trusted)
    with _POLICY_LOCK:
        previous_safe = os.environ.get(_TORCH_FORCE_SAFE)
        previous_unsafe = os.environ.get(_TORCH_FORCE_UNSAFE)
        token = None
        checkpoint_loader = None
        previous_local_loader: Any = _MISSING
        try:
            token = _TRUSTED_CHECKPOINTS.set(trusted)
            from mmengine.runner.checkpoint import CheckpointLoader

            checkpoint_loader = CheckpointLoader
            previous_local_loader = CheckpointLoader._schemes.get('', _MISSING)

            def load_from_baoiad_local(filename, map_location=None):
                return load_checkpoint(filename, map_location=map_location)

            CheckpointLoader.register_scheme(
                '', load_from_baoiad_local, force=True)
            if trusted:
                os.environ.pop(_TORCH_FORCE_SAFE, None)
                os.environ[_TORCH_FORCE_UNSAFE] = '1'
                warnings.warn(
                    'Trusted checkpoint loading is enabled and can execute Python code from '
                    'pickle data; use this only for an independently verified checkpoint.',
                    TrustedCheckpointWarning,
                    stacklevel=2,
                )
            else:
                os.environ.pop(_TORCH_FORCE_UNSAFE, None)
                os.environ[_TORCH_FORCE_SAFE] = '1'
            yield
        finally:
            try:
                if checkpoint_loader is not None:
                    if previous_local_loader is _MISSING:
                        checkpoint_loader._schemes.pop('', None)
                    else:
                        checkpoint_loader.register_scheme(
                            '', previous_local_loader, force=True)
            finally:
                try:
                    _restore_environment(_TORCH_FORCE_SAFE, previous_safe)
                    _restore_environment(_TORCH_FORCE_UNSAFE, previous_unsafe)
                finally:
                    if token is not None:
                        _TRUSTED_CHECKPOINTS.reset(token)


def load_checkpoint(
    path: str | os.PathLike[str],
    *,
    map_location: Any = 'cpu',
    trusted: bool | None = None,
) -> Any:
    """Load a local checkpoint safely, or with an explicit trusted opt-in."""
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    allow_pickle = (
        trusted_checkpoint_loading_enabled()
        if trusted is None
        else bool(trusted)
    )
    import torch

    with _POLICY_LOCK:
        if allow_pickle and not trusted_checkpoint_loading_enabled():
            warnings.warn(
                'Loading a trusted legacy checkpoint can execute Python code.',
                TrustedCheckpointWarning,
                stacklevel=2,
            )
        try:
            if checkpoint_path.suffix == '.safetensors':
                from safetensors.torch import load_file

                device = str(map_location) if map_location is not None else 'cpu'
                return load_file(str(checkpoint_path), device=device)

            with _restricted_safe_globals(torch):
                return torch.load(
                    str(checkpoint_path),
                    map_location=map_location,
                    weights_only=not allow_pickle,
                )
        except pickle.UnpicklingError as exc:
            if allow_pickle or not _is_restricted_unpickling_error(exc):
                raise CheckpointLoadError(
                    f'Checkpoint loading failed for {checkpoint_path}: '
                    f'{type(exc).__name__}: {exc}. The file may be corrupt or incompatible; '
                    'trusted loading does not repair malformed serialization.'
                ) from exc
            raise UnsafeCheckpointError(
                f'Safe checkpoint loading rejected {checkpoint_path}: '
                f'{type(exc).__name__}: {exc}. Prefer a tensor-only .pth or .safetensors '
                'file. Use --trusted-checkpoint only after independently verifying the file '
                'origin and integrity; that mode can execute code.'
            ) from exc
        except CheckpointError:
            raise
        except Exception as exc:
            raise CheckpointLoadError(
                f'Checkpoint loading failed for {checkpoint_path}: '
                f'{type(exc).__name__}: {exc}. The file may be corrupt, unreadable, or '
                'incompatible with this runtime; trusted loading does not repair such errors.'
            ) from exc

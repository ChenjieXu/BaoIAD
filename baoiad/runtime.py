"""Runtime policies shared by BaoIAD command-line tools and download helpers."""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping, MutableMapping
from urllib.parse import urlparse
from urllib.request import urlopen


_OFFLINE_ENV_VARS = (
    'BAOIAD_OFFLINE',
    'HF_HUB_OFFLINE',
    'TRANSFORMERS_OFFLINE',
    'HF_DATASETS_OFFLINE',
)
_TRUE_VALUES = frozenset({'1', 'true', 'yes', 'on'})


class OfflineModeError(RuntimeError):
    """Raised before an operation that requires network access in offline mode."""


def configure_offline_mode(
    enabled: bool,
    *,
    environ: MutableMapping[str, str] | None = None,
) -> None:
    """Enable BaoIAD's process-wide offline policy when explicitly requested.

    Passing ``False`` intentionally leaves the environment untouched so an
    existing user or scheduler policy is never weakened or overwritten.
    """
    if not enabled:
        return
    target = os.environ if environ is None else environ
    for name in _OFFLINE_ENV_VARS:
        target[name] = '1'


def is_offline_mode(environ: Mapping[str, str] | None = None) -> bool:
    """Return whether BaoIAD or a supported model hub is configured offline."""
    source = os.environ if environ is None else environ
    return any(str(source.get(name, '')).strip().lower() in _TRUE_VALUES for name in _OFFLINE_ENV_VARS)


def require_network(
    action: str,
    *,
    url: str | None = None,
    expected_path: str | os.PathLike[str] | None = None,
) -> None:
    """Fail before a download when the current process is explicitly offline."""
    if not is_offline_mode():
        return
    location = f' ({url})' if url else ''
    cache_hint = f' Expected local file: {os.fspath(expected_path)}.' if expected_path else ''
    raise OfflineModeError(
        f'Cannot {action}{location} while offline mode is enabled. '
        f'Provide the required file in the configured cache/path.{cache_hint} Or rerun '
        'without --offline and without offline environment variables.'
    )


def require_torchvision_weights(weights, *, action: str) -> None:
    """Require a cached Torchvision weight file before constructing offline."""
    if weights is None or not is_offline_mode():
        return

    import torch

    url = str(weights.url)
    filename = os.path.basename(urlparse(url).path)
    cached_path = os.path.join(torch.hub.get_dir(), 'checkpoints', filename)
    if not os.path.isfile(cached_path):
        require_network(action, url=url, expected_path=cached_path)


def download_url(
    url: str,
    destination: str | os.PathLike[str],
    *,
    action: str,
    timeout: float = 60,
) -> str:
    """Download one file without mutating process-wide socket settings."""
    target = os.path.abspath(os.fspath(destination))
    require_network(action, url=url, expected_path=target)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    partial = f'{target}.part'
    try:
        with urlopen(url, timeout=timeout) as response, open(partial, 'wb') as output:
            shutil.copyfileobj(response, output)
        os.replace(partial, target)
    except Exception:
        if os.path.exists(partial):
            os.remove(partial)
        raise
    return target

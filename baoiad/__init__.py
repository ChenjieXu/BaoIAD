"""BaoIAD: Unified Industrial Anomaly Detection Benchmark."""

import os

from baoiad.utils.compat import ensure_legacy_imgaug_compat

__version__ = '0.1.0'

# Data root resolution: prefer env var, fallback to data/ directory
BAOIAD_DATA_ROOT = os.environ.get(
    'BAOIAD_DATA_ROOT',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'),
)

# HF mirror is opt-in — only set if user explicitly requests it
if os.environ.get('BAOIAD_USE_MIRROR') == '1':
    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

# DEPRECATED SHIM: legacy imgaug-based paths still require these compatibility
# aliases during import. Remove this once the repo no longer depends on imgaug.
ensure_legacy_imgaug_compat()

from . import engine, evaluation, models, structures, visualization  # noqa: F401
from . import datasets  # noqa: F401

__all__ = [
    'datasets',
    'engine',
    'evaluation',
    'models',
    'structures',
    'visualization',
]

"""BaoIAD: Unified Industrial Anomaly Detection Benchmark."""

import os

import numpy as np

__version__ = '0.1.0'

# Data root resolution: prefer env var, fallback to data/ directory
BAOIAD_DATA_ROOT = os.environ.get(
    'BAOIAD_DATA_ROOT',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'),
)

# HF mirror is opt-in — only set if user explicitly requests it
if os.environ.get('BAOIAD_USE_MIRROR') == '1':
    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

# NumPy 2.x removed `np.sctypes`, but several legacy imgaug-based pipelines
# imported by the repo still expect it during module import.
if not hasattr(np, 'sctypes'):
    np.sctypes = {  # type: ignore[attr-defined]
        'int': [np.int8, np.int16, np.int32, np.int64],
        'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
        'float': [np.float16, np.float32, np.float64],
        'complex': [np.complex64, np.complex128],
        'others': [np.bool_, np.object_, np.str_, np.bytes_],
    }

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

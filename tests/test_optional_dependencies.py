"""Optional dependency and registry-boundary tests."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from baoiad import register_all_modules
from baoiad.optional import OptionalDependencyError
from baoiad.registry import MODELS

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _run_fresh_python(code: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPOSITORY_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_flow_detectors_have_actionable_missing_dependency_errors() -> None:
    register_all_modules()

    for detector_name in ("FastFlowDetector", "UFlowDetector"):
        detector = MODELS.get(detector_name)
        assert detector is not None
        if detector.__module__ == "baoiad.models.detectors":
            with pytest.raises(OptionalDependencyError, match=r"\.\[flow\]"):
                detector()


def test_lazy_registry_remains_usable_without_optional_modules() -> None:
    code = r"""
import importlib.abc
import sys

# Load the declared core stack before simulating absent extras. PyTorch probes
# optional package specs (for example pandas) while initializing TorchDynamo.
import mmcv  # noqa: F401
import mmengine  # noqa: F401
import torch  # noqa: F401
import torchvision  # noqa: F401

blocked = {
    'FrEIA', 'faiss', 'geomloss', 'groundingdino', 'imgaug', 'matplotlib',
    'open_clip', 'pandas', 'segment_anything', 'skimage',
}

class BlockOptionalModules(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise ModuleNotFoundError(fullname, name=fullname)
        return None

sys.meta_path.insert(0, BlockOptionalModules())
import baoiad
from baoiad.registry import (
    DATASETS,
    DATA_SAMPLERS,
    HOOKS,
    LOOPS,
    METRICS,
    MODELS,
    OPTIMIZERS,
    OPTIM_WRAPPERS,
    OPTIM_WRAPPER_CONSTRUCTORS,
    PARAM_SCHEDULERS,
    RUNNERS,
    TRANSFORMS,
    VISUALIZERS,
)

assert 'baoiad.models' not in sys.modules
assert 'baoiad.datasets' not in sys.modules
assert MODELS.get('PatchCore') is not None
assert MODELS.get('FastFlowDetector') is not None
assert MODELS.get('UFlowDetector') is not None
assert DATASETS.get('MVTecADDataset') is not None
assert TRANSFORMS.get('ResizeAD') is not None
assert DATA_SAMPLERS.get('PerEpochOrderSampler') is not None
assert METRICS.get('AnomalyDetectionMetric') is not None
assert LOOPS.get('ADTestLoop') is not None
assert HOOKS.get('MemoryBankHook') is not None
assert VISUALIZERS.get('ADVisualizer') is not None
assert OPTIMIZERS.get('StableAdamW') is not None
assert OPTIM_WRAPPERS.get('OptimWrapper') is not None
assert OPTIM_WRAPPER_CONSTRUCTORS.get('SimpleNetOptimWrapperConstructor') is not None
assert PARAM_SCHEDULERS.get('WarmCosineLR') is not None
assert RUNNERS.get('Runner') is not None
print('lazy registry boundary PASS')
"""
    result = _run_fresh_python(code)

    assert result.returncode == 0, result.stderr
    assert "lazy registry boundary PASS" in result.stdout.splitlines()


def test_specialized_configs_register_modules_in_fresh_process() -> None:
    code = r"""
import baoiad
import sys
from mmengine.config import Config
from baoiad.registry import DATASETS, MODELS

assert 'baoiad.models' not in sys.modules
assert 'baoiad.datasets' not in sys.modules

regad = Config.fromfile('configs/regad/regad_wrn50_256_mvtec_strict.py')
vitad = Config.fromfile('configs/vitad/vitad_256_mvtec_strict.py')
assert MODELS.get(regad.model.type) is not None
assert MODELS.get(vitad.model.type) is not None
assert DATASETS.get(vitad.train_dataloader.dataset.type) is not None
print('specialized config registration PASS')
"""
    result = _run_fresh_python(code)

    assert result.returncode == 0, result.stderr
    assert "specialized config registration PASS" in result.stdout.splitlines()

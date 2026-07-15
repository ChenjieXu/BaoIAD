"""Optional dependency and registry-boundary tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from baoiad import register_all_modules
from baoiad.optional import OptionalDependencyError
from baoiad.registry import MODELS

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_flow_detectors_have_actionable_missing_dependency_errors() -> None:
    register_all_modules()

    for detector_name in ("FastFlowDetector", "UFlowDetector"):
        detector = MODELS.get(detector_name)
        assert detector is not None
        if detector.__module__ == "baoiad.models.detectors":
            with pytest.raises(OptionalDependencyError, match=r"\.\[flow\]"):
                detector()


def test_patchcore_remains_registered_without_optional_modules() -> None:
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
from baoiad import register_all_modules
from baoiad.registry import MODELS
register_all_modules()
assert MODELS.get('PatchCore') is not None
assert MODELS.get('FastFlowDetector') is not None
assert MODELS.get('UFlowDetector') is not None
print('registry boundary PASS')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "registry boundary PASS" in result.stdout.splitlines()

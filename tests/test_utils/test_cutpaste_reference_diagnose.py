"""Tests for CutPaste reference diagnose helpers."""

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "cutpaste_reference_diagnose.py"
pytestmark = pytest.mark.optional
if not TOOL.is_file():
    pytest.skip(
        "legacy research-only diagnostic tool is excluded from the public release",
        allow_module_level=True,
    )


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "baoiad_cutpaste_reference_diagnose", TOOL
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_extract_prefixed_state_dict_strips_prefix():
    module = _load_module()
    state_dict = {
        "backbone.net.conv.weight": 1,
        "backbone.net.bn.weight": 2,
        "head.0.weight": 3,
    }

    extracted = module._extract_prefixed_state_dict(state_dict, "backbone.net.")

    assert extracted == {
        "conv.weight": 1,
        "bn.weight": 2,
    }


def test_resolve_alt_score_mode_defaults_to_backbone():
    module = _load_module()
    assert module._resolve_alt_score_mode("hazelnut") == "backbone_mahalanobis"
    assert module._resolve_alt_score_mode("carpet") == "backbone_mahalanobis"
    assert module._resolve_alt_score_mode("screw") == "classifier_prob"
    assert module._resolve_alt_score_mode("bottle") == "backbone_mahalanobis"


def test_score_gap_stats_splits_normal_and_anomaly():
    module = _load_module()

    stats = module._score_gap_stats([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])

    assert stats["normal"]["count"] == 2
    assert stats["anomaly"]["count"] == 2
    assert stats["score_gap_mean"] > 0

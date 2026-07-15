"""Tests for CutPaste density transfer diagnose helpers."""

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "cutpaste_density_transfer_diagnose.py"
pytestmark = pytest.mark.optional
if not TOOL.is_file():
    pytest.skip(
        "legacy research-only diagnostic tool is excluded from the public release",
        allow_module_level=True,
    )


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "baoiad_cutpaste_density_transfer_diagnose", TOOL
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_checkpoint_label_prefers_iter_token():
    module = _load_module()

    assert module._checkpoint_label("/tmp/iter_20.pth") == "iter_20"
    assert module._checkpoint_label("/tmp/epoch_2.pth") == "epoch_2"
    assert module._checkpoint_label("/tmp/model_final.pth") == "model_final"


def test_build_transfer_summary_reports_positive_gap_for_anomalies():
    module = _load_module()
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.7, 0.8]
    import torch

    train_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    test_embeddings = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])

    summary = module._build_transfer_summary(
        labels, scores, train_embeddings, test_embeddings
    )

    assert summary["metrics"]["image_auroc"] == 1.0
    assert summary["score_gap"]["score_gap_mean"] > 0
    assert summary["train_embeddings"]["shape"] == [2, 2]

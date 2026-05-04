"""Tests for CutPaste scoring diagnose helpers."""

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    path = ROOT / 'tools' / 'cutpaste_scoring_diagnose.py'
    spec = importlib.util.spec_from_file_location('baoiad_cutpaste_scoring_diagnose', path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_rank_samples_splits_normals_and_anomalies():
    module = _load_module()
    labels = np.array([0, 0, 1, 1, 0, 1], dtype=np.int64)
    scores = np.array([0.1, 0.9, 0.8, 0.2, 0.3, 0.7], dtype=np.float32)

    ranked = module._rank_samples(labels, scores, top_k=2)

    assert ranked['hardest_normals'] == [4, 1]
    assert ranked['easiest_normals'] == [0, 4]
    assert ranked['hardest_anomalies'] == [3, 5]
    assert ranked['easiest_anomalies'] == [5, 2]


def test_constant_score_maps_match_mask_shape():
    module = _load_module()
    scores = np.array([0.1, 0.9], dtype=np.float32)
    masks = np.zeros((2, 8, 8), dtype=np.float32)

    maps = module._constant_score_maps(scores, masks)

    assert maps.shape == masks.shape
    assert np.allclose(maps[0], 0.1)
    assert np.allclose(maps[1], 0.9)


def test_safe_probability_metric_returns_none_on_invalid_scores():
    module = _load_module()

    def _raise(_):
        raise ValueError('bad')

    result = module._safe_probability_metric(_raise, np.array([1.2], dtype=np.float32))

    assert result is None

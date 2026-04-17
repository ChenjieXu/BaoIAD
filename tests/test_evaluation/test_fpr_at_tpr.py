"""Tests for FPR@TPR metric."""

import numpy as np

from baoiad.evaluation.fpr_at_tpr import compute_fpr_at_tpr


class TestFPRAtTPR:
    def test_perfect_separation(self):
        gt = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])
        fpr = compute_fpr_at_tpr(gt, scores, 0.95)
        assert fpr == 0.0

    def test_matches_stepwise_roc_reference(self):
        gt = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.65, 0.5, 0.4, 0.7, 0.6, 0.55])
        assert compute_fpr_at_tpr(gt, scores, 0.95) == np.float64(1 / 3)
        assert compute_fpr_at_tpr(gt, scores, 0.50) == np.float64(1 / 3)

    def test_random(self):
        rng = np.random.default_rng(42)
        gt = rng.integers(0, 2, 100)
        scores = rng.random(100)
        fpr = compute_fpr_at_tpr(gt, scores, 0.95)
        assert 0.0 <= fpr <= 1.0

    def test_single_class(self):
        gt = np.zeros(10)
        scores = np.random.rand(10)
        fpr = compute_fpr_at_tpr(gt, scores, 0.95)
        assert fpr == 0.0

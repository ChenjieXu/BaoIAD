"""Tests for ECE metric."""

import numpy as np
import pytest

from baoiad.evaluation.ece import compute_ece


class TestECE:
    def test_perfect_calibration(self):
        # Scores match labels perfectly
        gt = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        ece = compute_ece(gt, scores)
        assert ece < 0.1

    def test_worst_calibration(self):
        gt = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        ece = compute_ece(gt, scores)
        assert ece > 0.3

    def test_constant_scores_match_prevalence(self):
        gt = np.array([0, 1, 0, 1])
        scores = np.array([0.5, 0.5, 0.5, 0.5])
        ece = compute_ece(gt, scores)
        assert ece == 0.0

    def test_constant_scores_keep_nonzero_error_when_prevalence_differs(self):
        gt = np.array([0, 0, 0, 1])
        scores = np.array([0.5, 0.5, 0.5, 0.5])
        ece = compute_ece(gt, scores, n_bins=2)
        assert ece == pytest.approx(0.25)

    def test_out_of_range_scores_raise(self):
        gt = np.array([0, 1, 0, 1])
        scores = np.array([-0.1, 0.4, 0.7, 1.2])
        with pytest.raises(ValueError, match=r'\[0, 1\]'):
            compute_ece(gt, scores)

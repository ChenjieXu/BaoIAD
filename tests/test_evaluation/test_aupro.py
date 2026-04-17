"""Tests for AUPRO metric (anomalib-aligned implementation)."""

import numpy as np
import pytest

from baoiad.evaluation.aupro import compute_aupro


class TestAUPRO:
    def test_matches_manual_reference_curve(self):
        """Simple 1-region example with hand-computed AUPRO = 0.75."""
        gt = np.array([[[1, 1, 0, 0]]], dtype=np.float32)
        pred = np.array([[[0.9, 0.8, 0.85, 0.1]]], dtype=np.float32)
        result = compute_aupro(gt, pred, max_fpr=1.0)
        assert result == pytest.approx(0.75)

    def test_perfect_segmentation(self):
        """Perfect segmentation should yield high AUPRO."""
        np.random.seed(42)
        gt = np.zeros((4, 32, 32), dtype=np.float32)
        gt[2:, 8:24, 8:24] = 1.0
        pred = np.random.rand(4, 32, 32).astype(np.float32) * 0.3
        pred[2:, 8:24, 8:24] += 0.7
        result = compute_aupro(gt, pred)
        assert result > 0.5

    def test_empty_masks(self):
        """Empty masks (no anomaly regions) should return 0."""
        gt = np.zeros((4, 32, 32), dtype=np.float32)
        pred = np.random.rand(4, 32, 32).astype(np.float32)
        result = compute_aupro(gt, pred)
        assert result == 0.0

    def test_no_background(self):
        """All anomaly (no background) should return 0."""
        gt = np.ones((2, 16, 16), dtype=np.float32)
        pred = np.random.rand(2, 16, 16).astype(np.float32)
        result = compute_aupro(gt, pred)
        assert result == 0.0

    def test_single_region(self):
        """Single region with perfect prediction."""
        gt = np.zeros((1, 32, 32), dtype=np.float32)
        gt[0, 8:24, 8:24] = 1.0
        pred = np.zeros((1, 32, 32), dtype=np.float32)
        pred[0, 8:24, 8:24] = 1.0
        result = compute_aupro(gt, pred)
        assert result > 0.9  # Should be close to 1.0

    def test_max_fpr_parameter(self):
        """Higher max_fpr should give higher or equal AUPRO."""
        np.random.seed(42)
        gt = np.zeros((4, 32, 32), dtype=np.float32)
        gt[2:, 8:24, 8:24] = 1.0
        pred = np.random.rand(4, 32, 32).astype(np.float32) * 0.3
        pred[2:, 8:24, 8:24] += 0.7

        result_03 = compute_aupro(gt, pred, max_fpr=0.3)
        result_05 = compute_aupro(gt, pred, max_fpr=0.5)
        # Allow small tolerance for floating point comparison
        assert result_05 >= result_03 - 1e-6

    def test_multiple_regions_per_image(self):
        """Multiple disconnected regions in same image."""
        gt = np.zeros((1, 32, 32), dtype=np.float32)
        gt[0, 4:10, 4:10] = 1.0
        gt[0, 20:28, 20:28] = 1.0
        pred = np.zeros((1, 32, 32), dtype=np.float32)
        pred[0, 4:10, 4:10] = 0.9
        pred[0, 20:28, 20:28] = 0.9
        result = compute_aupro(gt, pred)
        assert result > 0.5

    def test_multiple_images_with_regions(self):
        """Multiple images, each with anomaly regions."""
        np.random.seed(42)
        gt = np.zeros((3, 32, 32), dtype=np.float32)
        gt[0, 4:12, 4:12] = 1.0
        gt[1, 16:24, 8:16] = 1.0
        gt[2, 8:20, 16:28] = 1.0

        pred = np.random.rand(3, 32, 32).astype(np.float32) * 0.3
        pred[0, 4:12, 4:12] = 0.8
        pred[1, 16:24, 8:16] = 0.9
        pred[2, 8:20, 16:28] = 0.85

        result = compute_aupro(gt, pred)
        assert result > 0.5

    def test_shape_mismatch(self):
        """Shape mismatch should raise ValueError."""
        gt = np.zeros((2, 16, 16), dtype=np.float32)
        pred = np.zeros((2, 32, 32), dtype=np.float32)
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_aupro(gt, pred)

    def test_result_in_valid_range(self):
        """Result should always be in [0, 1]."""
        np.random.seed(42)
        gt = np.random.randint(0, 2, (4, 32, 32)).astype(np.float32)
        pred = np.random.rand(4, 32, 32).astype(np.float32)
        result = compute_aupro(gt, pred)
        assert 0.0 <= result <= 1.0

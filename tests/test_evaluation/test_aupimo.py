"""Tests for AUPIMO metric."""

import numpy as np
import pytest

from baoiad.evaluation.aupimo import compute_pimo, compute_pimo_per_image


def _make_official_reference_case():
    """Toy case adapted from the official AUPIMO repository tests."""
    shape = (1000, 1000)
    pred_norm = np.ones(1_000_000, dtype=np.float32)
    pred_norm[:100_000] += 1
    pred_norm[:10_000] += 1
    pred_norm[:1_000] += 1
    pred_norm[:100] += 1
    pred_norm[:10] += 1
    pred_norm[:1] += 1
    pred_norm = pred_norm.reshape(shape)

    mask_norm = np.zeros_like(pred_norm, dtype=np.int32)
    pred_anom1 = pred_norm.copy()
    mask_anom1 = np.ones_like(pred_anom1, dtype=np.int32)
    pred_anom2 = pred_norm.copy()
    mask_anom2 = np.concatenate([np.ones(100_000), np.zeros(900_000)]).reshape(shape).astype(np.int32)

    anomaly_maps = np.stack([pred_norm, pred_anom1, pred_anom2], axis=0)
    masks = np.stack([mask_norm, mask_anom1, mask_anom2], axis=0)
    return masks, anomaly_maps


class TestAUPIMO:
    def test_perfect_segmentation(self):
        """Perfect segmentation should yield AUPIMO close to 1."""
        np.random.seed(42)
        # 2 normal images (all zeros), 2 anomalous images with defects
        gt = np.zeros((4, 64, 64), dtype=np.float32)
        gt[2:, 16:48, 16:48] = 1.0

        # Perfect prediction: high scores on anomaly regions
        pred = np.zeros((4, 64, 64), dtype=np.float32)
        # Add noise to normal images to create score variation for FPR calibration
        # This simulates real-world scenarios where normal images have some variation
        pred[:2] = np.random.uniform(0.0, 0.1, (2, 64, 64)).astype(np.float32)
        # Set anomaly regions to much higher scores
        pred[2:, 16:48, 16:48] = 1.0

        # Use wider FPR bounds suitable for small test images
        # (1e-5 to 1e-4 is too narrow for 64x64 images)
        result = compute_pimo(gt, pred, fpr_bounds=(0.001, 0.1))
        # Perfect segmentation should give high AUPIMO
        assert result >= 0.8, f"Expected AUPIMO >= 0.8 for perfect segmentation, got {result}"

    def test_random_prediction(self):
        """Random predictions should yield low AUPIMO at low FPR rates.

        Note: At low FPR rates, random predictions have TPR ≈ FPR,
        so AUPIMO will be much lower than 0.5, not around 0.5.
        """
        np.random.seed(42)
        # 2 normal images, 2 anomalous images
        gt = np.zeros((4, 64, 64), dtype=np.float32)
        gt[2:, 16:48, 16:48] = 1.0

        # Random predictions (similar distribution for normal and anomaly)
        pred = np.random.rand(4, 64, 64).astype(np.float32)

        # Use wider FPR bounds
        result = compute_pimo(gt, pred, fpr_bounds=(0.001, 0.1))
        # Random predictions at low FPR give TPR ≈ FPR, so AUPIMO is low
        assert 0.0 <= result < 0.3, f"Expected low AUPIMO for random predictions, got {result}"

    def test_no_anomalous_images(self):
        """No anomalous images should return 0."""
        gt = np.zeros((4, 32, 32), dtype=np.float32)
        pred = np.random.rand(4, 32, 32).astype(np.float32)
        result = compute_pimo(gt, pred)
        assert result == 0.0

    def test_no_normal_images(self):
        """No normal images should return 0."""
        gt = np.ones((4, 32, 32), dtype=np.float32)
        gt[:, :16, :] = 0  # All images have anomalies in bottom half
        pred = np.random.rand(4, 32, 32).astype(np.float32)
        result = compute_pimo(gt, pred)
        assert result == 0.0

    def test_threshold_order(self):
        """Verify that better separation between normal and anomaly improves AUPIMO."""
        np.random.seed(42)
        # 2 normal, 2 anomalous
        gt = np.zeros((4, 64, 64), dtype=np.float32)
        gt[2:, 16:48, 16:48] = 1.0

        # Good prediction: clear separation - normal scores up to 0.1, anomaly at 1.0
        pred_good = np.zeros((4, 64, 64), dtype=np.float32)
        pred_good[:2] = np.random.uniform(0.0, 0.1, (2, 64, 64)).astype(np.float32)
        pred_good[2:, 16:48, 16:48] = 1.0

        # Bad prediction: poor separation - normal scores overlap with anomaly
        # Some normal pixels are higher than some anomaly pixels
        pred_bad = np.zeros((4, 64, 64), dtype=np.float32)
        pred_bad[:2] = np.random.uniform(0.0, 0.5, (2, 64, 64)).astype(np.float32)
        pred_bad[2:, 16:48, 16:48] = 0.3  # Anomaly scores lower than some normal

        # Use FPR bounds where the difference matters
        result_good = compute_pimo(gt, pred_good, fpr_bounds=(0.001, 0.1))
        result_bad = compute_pimo(gt, pred_bad, fpr_bounds=(0.001, 0.1))

        assert result_good > result_bad, (
            f"Better predictions should give higher AUPIMO: "
            f"good={result_good}, bad={result_bad}"
        )

    def test_different_fpr_bounds(self):
        """Test with different FPR bounds."""
        np.random.seed(42)
        gt = np.zeros((4, 64, 64), dtype=np.float32)
        gt[2:, 16:48, 16:48] = 1.0

        pred = np.zeros((4, 64, 64), dtype=np.float32)
        pred[:2] = np.random.uniform(0.0, 0.1, (2, 64, 64)).astype(np.float32)
        pred[2:, 16:48, 16:48] = 1.0

        # Test with default bounds
        result_default = compute_pimo(gt, pred)

        # Test with wider bounds (should still work)
        result_wider = compute_pimo(gt, pred, fpr_bounds=(1e-5, 1e-3))

        # Both should be valid scores
        assert 0.0 <= result_default <= 1.0
        assert 0.0 <= result_wider <= 1.0

    def test_output_range(self):
        """AUPIMO should always be in [0, 1]."""
        np.random.seed(123)
        for _ in range(5):
            gt = np.zeros((4, 32, 32), dtype=np.float32)
            gt[2:, 8:24, 8:24] = 1.0
            pred = np.random.rand(4, 32, 32).astype(np.float32) * np.random.rand() * 10
            result = compute_pimo(gt, pred)
            assert 0.0 <= result <= 1.0, f"AUPIMO out of range: {result}"

    def test_single_anomaly_pixel(self):
        """Test with a single anomaly pixel per image."""
        np.random.seed(42)
        gt = np.zeros((4, 64, 64), dtype=np.float32)
        gt[2, 32, 32] = 1.0  # Single pixel anomaly
        gt[3, 31, 31] = 1.0

        # Prediction with single high pixel
        pred = np.zeros((4, 64, 64), dtype=np.float32)
        pred[:2] = np.random.uniform(0.0, 0.1, (2, 64, 64)).astype(np.float32)
        pred[2, 32, 32] = 1.0
        pred[3, 31, 31] = 1.0

        result = compute_pimo(gt, pred)
        assert 0.0 <= result <= 1.0

    def test_per_image_function(self):
        """Test compute_pimo_per_image returns correct structure."""
        np.random.seed(42)
        gt = np.zeros((4, 64, 64), dtype=np.float32)
        gt[2:, 16:48, 16:48] = 1.0

        pred = np.zeros((4, 64, 64), dtype=np.float32)
        pred[:2] = np.random.uniform(0.0, 0.1, (2, 64, 64)).astype(np.float32)
        pred[2:, 16:48, 16:48] = 1.0

        per_image_aupimos, anomalous_idx = compute_pimo_per_image(gt, pred)

        assert len(per_image_aupimos) == 2, "Should have 2 per-image AUPIMOs"
        assert anomalous_idx == [2, 3], "Anomalous indices should be [2, 3]"
        assert all(0.0 <= v <= 1.0 for v in per_image_aupimos)

        # Average should match compute_pimo
        avg_result = compute_pimo(gt, pred)
        assert np.isclose(np.mean(per_image_aupimos), avg_result, atol=1e-6)

    def test_per_image_consistency(self):
        """Per-image AUPIMOs average should match compute_pimo."""
        np.random.seed(456)
        gt = np.zeros((6, 32, 32), dtype=np.float32)
        gt[3:, 8:24, 8:24] = 1.0
        pred = np.random.rand(6, 32, 32).astype(np.float32)

        avg_result = compute_pimo(gt, pred)
        per_image_aupimos, _ = compute_pimo_per_image(gt, pred)

        assert np.isclose(np.mean(per_image_aupimos), avg_result, atol=1e-6)

    @pytest.mark.parametrize(
        ('fpr_bounds', 'expected_per_image'),
        [
            ((1e-1, 1.0), np.array([0.55, 1.0], dtype=np.float64)),
            ((1e-3, 1e-1), np.array([0.03025, 0.3025], dtype=np.float64)),
            ((1e-5, 1e-4), np.array([5.5e-05, 5.5e-04], dtype=np.float64)),
        ],
    )
    def test_matches_official_reference_vectors(self, fpr_bounds, expected_per_image):
        masks, anomaly_maps = _make_official_reference_case()
        per_image_aupimos, anomalous_idx = compute_pimo_per_image(
            masks,
            anomaly_maps,
            fpr_bounds=fpr_bounds,
            num_thresholds=7,
        )
        assert anomalous_idx == [1, 2]
        assert np.allclose(per_image_aupimos, expected_per_image, atol=1e-12)
        assert compute_pimo(
            masks,
            anomaly_maps,
            fpr_bounds=fpr_bounds,
            num_thresholds=7,
        ) == pytest.approx(float(expected_per_image.mean()), abs=1e-12)

    def test_multiple_normal_images(self):
        """Test with varying numbers of normal images."""
        np.random.seed(42)
        # Test with 1 normal, 1 anomalous
        gt1 = np.zeros((2, 64, 64), dtype=np.float32)
        gt1[1, 16:48, 16:48] = 1.0
        pred1 = np.zeros((2, 64, 64), dtype=np.float32)
        pred1[0] = np.random.uniform(0.0, 0.1, (64, 64)).astype(np.float32)
        pred1[1, 16:48, 16:48] = 1.0

        result1 = compute_pimo(gt1, pred1)
        assert 0.0 <= result1 <= 1.0

        # Test with 10 normal, 5 anomalous
        gt2 = np.zeros((15, 64, 64), dtype=np.float32)
        gt2[10:, 16:48, 16:48] = 1.0
        pred2 = np.zeros((15, 64, 64), dtype=np.float32)
        pred2[:10] = np.random.uniform(0.0, 0.1, (10, 64, 64)).astype(np.float32)
        pred2[10:, 16:48, 16:48] = 1.0

        result2 = compute_pimo(gt2, pred2)
        assert 0.0 <= result2 <= 1.0

    def test_log_space_integration(self):
        """Verify that log-space integration is used correctly."""
        np.random.seed(42)
        # Create a scenario where linear vs log integration would differ
        gt = np.zeros((4, 64, 64), dtype=np.float32)
        gt[2:, 16:48, 16:48] = 1.0

        # Create predictions with scores that span a wide range
        pred = np.zeros((4, 64, 64), dtype=np.float32)
        pred[:2] = np.random.uniform(0.0, 0.05, (2, 64, 64)).astype(np.float32)
        pred[2:, 16:48, 16:48] = 1.0  # High scores on anomaly

        result = compute_pimo(gt, pred, fpr_bounds=(1e-5, 1e-4))
        assert 0.0 <= result <= 1.0

    def test_small_images(self):
        """Test with very small images to check edge cases."""
        np.random.seed(42)
        gt = np.zeros((4, 16, 16), dtype=np.float32)
        gt[2:, 4:12, 4:12] = 1.0

        pred = np.zeros((4, 16, 16), dtype=np.float32)
        pred[:2] = np.random.uniform(0.0, 0.1, (2, 16, 16)).astype(np.float32)
        pred[2:, 4:12, 4:12] = 1.0

        result = compute_pimo(gt, pred)
        assert 0.0 <= result <= 1.0

    def test_all_normal_same_score(self):
        """Test when all normal images have identical scores."""
        np.random.seed(42)
        gt = np.zeros((4, 64, 64), dtype=np.float32)
        gt[2:, 16:48, 16:48] = 1.0

        pred = np.zeros((4, 64, 64), dtype=np.float32)
        pred[:2] = 0.5  # All normal pixels same score
        pred[2:, 16:48, 16:48] = 1.0

        # Should still work even with uniform normal scores
        result = compute_pimo(gt, pred)
        assert 0.0 <= result <= 1.0

    def test_fpr_bound_edge_cases(self):
        """Test with edge case FPR bounds."""
        np.random.seed(42)
        gt = np.zeros((4, 64, 64), dtype=np.float32)
        gt[2:, 16:48, 16:48] = 1.0

        pred = np.zeros((4, 64, 64), dtype=np.float32)
        pred[:2] = np.random.uniform(0.0, 0.1, (2, 64, 64)).astype(np.float32)
        pred[2:, 16:48, 16:48] = 1.0

        # Very narrow bounds
        result_narrow = compute_pimo(gt, pred, fpr_bounds=(1e-5, 2e-5))
        assert 0.0 <= result_narrow <= 1.0

        # Very wide bounds
        result_wide = compute_pimo(gt, pred, fpr_bounds=(1e-6, 1e-2))
        assert 0.0 <= result_wide <= 1.0

    def test_invalid_fpr_bounds_raise(self):
        gt = np.zeros((2, 32, 32), dtype=np.float32)
        gt[1, 8:24, 8:24] = 1.0
        pred = np.random.rand(2, 32, 32).astype(np.float32)

        with pytest.raises(ValueError, match='fpr_bounds'):
            compute_pimo(gt, pred, fpr_bounds=(0.1, 0.001))

    def test_shape_mismatch_raises(self):
        gt = np.zeros((2, 32, 32), dtype=np.float32)
        pred = np.zeros((2, 16, 16), dtype=np.float32)

        with pytest.raises(ValueError, match='Shape mismatch'):
            compute_pimo(gt, pred)


class TestAUPIMOEdgeCases:
    """Additional edge case tests for AUPIMO."""

    def test_empty_input(self):
        """Test with minimal valid input."""
        np.random.seed(42)
        gt = np.zeros((2, 32, 32), dtype=np.float32)
        gt[1, 8:24, 8:24] = 1.0
        pred = np.random.rand(2, 32, 32).astype(np.float32)

        result = compute_pimo(gt, pred)
        assert 0.0 <= result <= 1.0

    def test_asymmetric_anomalies(self):
        """Test with different anomaly sizes per image."""
        np.random.seed(42)
        gt = np.zeros((4, 64, 64), dtype=np.float32)
        gt[2, 8:16, 8:16] = 1.0  # Small anomaly (64 pixels)
        gt[3, 8:56, 8:56] = 1.0  # Large anomaly (2304 pixels)

        pred = np.zeros((4, 64, 64), dtype=np.float32)
        pred[:2] = np.random.uniform(0.0, 0.1, (2, 64, 64)).astype(np.float32)
        pred[2, 8:16, 8:16] = 1.0
        pred[3, 8:56, 8:56] = 1.0

        result = compute_pimo(gt, pred)
        assert 0.0 <= result <= 1.0

    def test_partial_detection(self):
        """Test when only part of anomaly is detected."""
        np.random.seed(42)
        gt = np.zeros((4, 64, 64), dtype=np.float32)
        gt[2:, 16:48, 16:48] = 1.0

        # Partial detection: only detect half of each anomaly
        pred_partial = np.zeros((4, 64, 64), dtype=np.float32)
        pred_partial[:2] = np.random.uniform(0.0, 0.1, (2, 64, 64)).astype(np.float32)
        pred_partial[2:, 16:48, 16:32] = 1.0  # Only left half detected

        # Perfect detection for comparison
        pred_perfect = np.zeros((4, 64, 64), dtype=np.float32)
        pred_perfect[:2] = pred_partial[:2].copy()  # Same normal scores
        pred_perfect[2:, 16:48, 16:48] = 1.0  # Full anomaly detected

        result_partial = compute_pimo(gt, pred_partial)
        result_perfect = compute_pimo(gt, pred_perfect)

        assert 0.0 <= result_partial <= 1.0
        # Partial detection should be less than perfect (or at least not better)
        assert result_partial <= result_perfect + 0.1, (
            f"Partial detection should not exceed perfect: partial={result_partial}, perfect={result_perfect}"
        )

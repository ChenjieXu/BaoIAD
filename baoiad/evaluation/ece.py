"""ECE metric: Expected Calibration Error for anomaly probabilities."""

import numpy as np


def _validate_probabilities(pred_scores: np.ndarray) -> np.ndarray:
    """Validate probability-like scores used for calibration metrics."""
    scores = np.asarray(pred_scores, dtype=np.float64)
    if scores.size == 0:
        return scores
    if not np.isfinite(scores).all():
        raise ValueError('pred_scores must be finite for ECE computation.')
    if ((scores < 0.0) | (scores > 1.0)).any():
        raise ValueError('pred_scores must already be probabilities in [0, 1] for ECE.')
    return scores


def compute_ece(gt_labels: np.ndarray, pred_scores: np.ndarray, n_bins: int = 15) -> float:
    """Compute Expected Calibration Error.

    ECE assumes ``pred_scores`` are already calibrated probabilities. The metric
    intentionally does not rescale arbitrary anomaly scores, because per-batch
    min-max normalization changes the probability semantics being measured.

    Args:
        gt_labels: (N,) binary ground truth labels.
        pred_scores: (N,) predicted probabilities in [0, 1].
        n_bins: Number of bins for calibration.

    Returns:
        ECE value (lower is better).
    """
    scores = _validate_probabilities(pred_scores)
    if scores.size == 0:
        return 0.0

    labels = (np.asarray(gt_labels, dtype=np.float64) > 0.5).astype(np.float64)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(scores)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            mask = (scores >= lo) & (scores <= hi)
        else:
            mask = (scores >= lo) & (scores < hi)
        n_in_bin = int(mask.sum())
        if n_in_bin == 0:
            continue

        avg_confidence = float(scores[mask].mean())
        avg_accuracy = float(labels[mask].mean())
        ece += (n_in_bin / total) * abs(avg_accuracy - avg_confidence)

    return float(ece)


def compute_pixel_ece(gt_masks: np.ndarray | list[np.ndarray], pred_maps: np.ndarray | list[np.ndarray], n_bins: int = 15) -> float:
    """Compute pixel-level ECE for probability maps."""
    if isinstance(gt_masks, np.ndarray):
        gt_flat = gt_masks.ravel()
    else:
        gt_flat = np.concatenate([np.asarray(mask).reshape(-1) for mask in gt_masks])

    if isinstance(pred_maps, np.ndarray):
        pred_flat = pred_maps.ravel()
    else:
        pred_flat = np.concatenate([np.asarray(pred_map).reshape(-1) for pred_map in pred_maps])

    return compute_ece(gt_flat, pred_flat, n_bins)

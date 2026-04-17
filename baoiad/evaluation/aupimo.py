"""AUPIMO metric: Area Under the Per-Image Missed Overlap curve.

Reference: "AUPIMO: Redefining Visual Anomaly Detection Benchmarks
with High Speed and Low Tolerance" (BMVC 2024)
https://arxiv.org/abs/2401.01984
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid


def _validate_pimo_inputs(
    gt_masks: np.ndarray,
    pred_maps: np.ndarray,
    fpr_bounds: tuple[float, float],
    num_thresholds: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate common AUPIMO inputs."""
    gt_masks = np.asarray(gt_masks)
    pred_maps = np.asarray(pred_maps, dtype=np.float64)

    if gt_masks.shape != pred_maps.shape:
        raise ValueError(f'Shape mismatch: gt_masks {gt_masks.shape} vs pred_maps {pred_maps.shape}')
    if gt_masks.ndim != 3:
        raise ValueError(
            f'AUPIMO expects (N, H, W) inputs, got gt_masks.ndim={gt_masks.ndim}.'
        )

    fpr_lo, fpr_hi = fpr_bounds
    if not (0.0 < fpr_lo < fpr_hi <= 1.0):
        raise ValueError(
            f'fpr_bounds must satisfy 0 < lower < upper <= 1, got {fpr_bounds}.'
        )
    if num_thresholds < 2:
        raise ValueError('num_thresholds must be at least 2.')

    return gt_masks, pred_maps


def _build_normal_only_thresholds(
    normal_scores_flat: np.ndarray,
    fpr_bounds: tuple[float, float],
    num_thresholds: int,
) -> np.ndarray:
    """Build an ascending threshold grid from normal-image scores only."""
    _fpr_lo, _fpr_hi = fpr_bounds
    min_score = float(normal_scores_flat.min())
    max_score = float(normal_scores_flat.max())
    if num_thresholds == 2:
        return np.asarray([min_score, max_score], dtype=np.float64)
    return np.linspace(min_score, max_score, num_thresholds, dtype=np.float64)


def _compute_shared_fprs(pred_maps: np.ndarray, normal_idx: list[int], thresholds: np.ndarray) -> np.ndarray:
    """Compute shared FPR values for a threshold grid."""
    normal_maps = pred_maps[normal_idx]
    shared_fprs = []
    for threshold in thresholds:
        image_fprs = (normal_maps >= threshold).mean(axis=(1, 2))
        shared_fprs.append(float(image_fprs.mean()))
    return np.asarray(shared_fprs, dtype=np.float64)


def _index_at_shared_fpr_level(shared_fprs: np.ndarray, fpr_level: float) -> int:
    """Match the official AUPIMO threshold selection at a shared-FPR level."""
    shared_fprs = np.asarray(shared_fprs, dtype=np.float64)
    shared_fpr_min = float(shared_fprs.min())
    shared_fpr_max = float(shared_fprs.max())
    if fpr_level < shared_fpr_min or fpr_level > shared_fpr_max:
        raise ValueError(
            "Requested fpr_level is outside the shared_fpr range "
            f"[{shared_fpr_min}, {shared_fpr_max}]: {fpr_level}."
        )

    if fpr_level == 0.0:
        return int(np.min(np.where(shared_fprs == fpr_level)))
    if fpr_level == 1.0:
        return int(np.max(np.where(shared_fprs == fpr_level)))
    return int(np.argmin(np.abs(shared_fprs - fpr_level)))


def _integrate_curve_in_log_fpr(
    shared_fprs: np.ndarray,
    image_tprs: np.ndarray,
    fpr_bounds: tuple[float, float],
) -> float:
    """Integrate a per-image PIMO curve over log-FPR."""
    fpr_lo, fpr_hi = fpr_bounds
    try:
        fpr_lo_idx = _index_at_shared_fpr_level(shared_fprs, fpr_lo)
        fpr_hi_idx = _index_at_shared_fpr_level(shared_fprs, fpr_hi)
    except ValueError:
        return 0.0

    thresh_lower_bound_idx = fpr_hi_idx
    thresh_upper_bound_idx = fpr_lo_idx
    if thresh_lower_bound_idx >= thresh_upper_bound_idx:
        return 0.0

    bounded_fprs = np.flip(shared_fprs[thresh_lower_bound_idx : thresh_upper_bound_idx + 1])
    bounded_tprs = np.flip(image_tprs[thresh_lower_bound_idx : thresh_upper_bound_idx + 1])

    valid = np.isfinite(np.log(bounded_fprs))
    if not valid.any():
        return 0.0
    bounded_fprs = bounded_fprs[valid]
    bounded_tprs = bounded_tprs[valid]
    if bounded_fprs.size < 2:
        return 0.0

    area = trapezoid(bounded_tprs, np.log(bounded_fprs))
    norm = np.log(fpr_hi) - np.log(fpr_lo)
    if norm <= 0:
        return 0.0
    return float(np.clip(area / norm, 0.0, 1.0))


def _prepare_pimo_curves(
    gt_masks: np.ndarray,
    pred_maps: np.ndarray,
    fpr_bounds: tuple[float, float],
    num_thresholds: int,
) -> tuple[np.ndarray, list[int], np.ndarray, np.ndarray] | tuple[None, list[int], None, None]:
    """Prepare shared-FPR samples and per-image TPR curves for AUPIMO."""
    gt_masks, pred_maps = _validate_pimo_inputs(gt_masks, pred_maps, fpr_bounds, num_thresholds)

    normal_idx = [i for i in range(len(gt_masks)) if gt_masks[i].max() == 0]
    anomalous_idx = [i for i in range(len(gt_masks)) if gt_masks[i].max() > 0]

    if not normal_idx or not anomalous_idx:
        return None, anomalous_idx, None, None

    normal_scores_flat = pred_maps[normal_idx].reshape(-1)
    thresholds = _build_normal_only_thresholds(normal_scores_flat, fpr_bounds, num_thresholds)
    if thresholds.size < 2:
        return None, anomalous_idx, None, None

    shared_fprs = _compute_shared_fprs(pred_maps, normal_idx, thresholds)

    per_image_tprs = []
    for img_idx in anomalous_idx:
        gt_mask = gt_masks[img_idx] > 0
        pred_map = pred_maps[img_idx]
        anomaly_pixels = float(gt_mask.sum())
        if anomaly_pixels == 0:
            per_image_tprs.append(np.zeros_like(shared_fprs))
            continue

        image_tprs = []
        for threshold in thresholds:
            tp = ((pred_map >= threshold) & gt_mask).sum()
            image_tprs.append(float(tp / anomaly_pixels))
        per_image_tprs.append(np.asarray(image_tprs, dtype=np.float64))

    return shared_fprs, anomalous_idx, np.asarray(per_image_tprs), thresholds


def compute_pimo(
    gt_masks: np.ndarray,
    pred_maps: np.ndarray,
    fpr_bounds: tuple[float, float] = (1e-5, 1e-4),
    num_thresholds: int = 300,
) -> float:
    """Compute AUPIMO (Area Under Per-Image Missed Overlap).

    This implementation follows the official AUPIMO paper definition:
    1. Compute Shared FPR (F_sh) from normal images only
    2. Compute Per-Image TPR (T_i) for each anomalous image
    3. Integrate in log(F_sh) space
    4. Return the average of per-image AUPIMOs

    Key differences from the old implementation:
    - FPR is computed from normal images only (not all normal pixels)
    - Uses Image-level TPR instead of region-level overlap
    - Integrates in log-space instead of linear space
    - Returns average of per-image AUPIMOs instead of AUC of averaged curve

    Args:
        gt_masks: (N, H, W) binary ground truth masks.
        pred_maps: (N, H, W) anomaly score maps.
        fpr_bounds: (lower, upper) FPR range for integration.
            Default (1e-5, 1e-4) corresponds to 0.001% to 0.01% FPR.
        num_thresholds: Number of thresholds to evaluate.

    Returns:
        AUPIMO value (average of per-image AUPIMOs).
    """
    shared_fprs, anomalous_idx, per_image_tprs, _thresholds = _prepare_pimo_curves(
        gt_masks, pred_maps, fpr_bounds, num_thresholds
    )
    if shared_fprs is None or per_image_tprs is None or not anomalous_idx:
        return 0.0

    per_image_aupimos = [
        _integrate_curve_in_log_fpr(shared_fprs, image_tprs, fpr_bounds)
        for image_tprs in per_image_tprs
    ]
    return float(np.mean(per_image_aupimos)) if per_image_aupimos else 0.0


def compute_pimo_per_image(
    gt_masks: np.ndarray,
    pred_maps: np.ndarray,
    fpr_bounds: tuple[float, float] = (1e-5, 1e-4),
    num_thresholds: int = 300,
) -> tuple[np.ndarray, list[int]]:
    """Compute per-image AUPIMO values.

    This function returns individual AUPIMO scores for each anomalous image,
    useful for detailed analysis.

    Args:
        gt_masks: (N, H, W) binary ground truth masks.
        pred_maps: (N, H, W) anomaly score maps.
        fpr_bounds: (lower, upper) FPR range for integration.
        num_thresholds: Number of thresholds to evaluate.

    Returns:
        Tuple of:
            - Array of per-image AUPIMO values
            - List of anomalous image indices
    """
    shared_fprs, anomalous_idx, per_image_tprs, _thresholds = _prepare_pimo_curves(
        gt_masks, pred_maps, fpr_bounds, num_thresholds
    )
    if shared_fprs is None or per_image_tprs is None or not anomalous_idx:
        return np.array([]), []

    per_image_aupimos = [
        _integrate_curve_in_log_fpr(shared_fprs, image_tprs, fpr_bounds)
        for image_tprs in per_image_tprs
    ]
    return np.asarray(per_image_aupimos, dtype=np.float64), anomalous_idx

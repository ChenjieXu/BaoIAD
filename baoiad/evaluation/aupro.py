"""AUPRO metric: Area Under the Per-Region Overlap curve.

Implementation aligned with anomalib's global sorting method for better accuracy.
"""

import numpy as np
from scipy.ndimage import label as connected_components
from sklearn.metrics import auc


def _make_global_region_labels(cca: np.ndarray) -> np.ndarray:
    """Offset connected component labels across batch to make them unique.

    Args:
        cca: (N, H, W) integer labels, starting at 0 for each image.

    Returns:
        (N, H, W) labels where 0 is background, positive labels are unique across batch.
    """
    cca_off = cca.copy()
    current_offset = 0

    for i in range(len(cca)):
        img_labels = cca_off[i]
        unique_fg = np.unique(img_labels[img_labels > 0])
        if len(unique_fg) == 0:
            continue
        # Shift foreground labels
        fg_mask = img_labels > 0
        img_labels[fg_mask] = img_labels[fg_mask] + current_offset
        cca_off[i] = img_labels
        current_offset += int(len(unique_fg))

    return cca_off


def compute_aupro(gt_masks: np.ndarray, pred_maps: np.ndarray, max_fpr: float = 0.3) -> float:
    """Compute AUPRO using global sorting method (anomalib-aligned).

    This implementation uses a global sorting approach which is:
    - More accurate (based on actual pixel ordering)
    - More efficient (O(N log N) vs O(T * N * R) for threshold-based)

    Args:
        gt_masks: (N, H, W) binary ground truth masks.
        pred_maps: (N, H, W) anomaly score maps.
        max_fpr: Maximum false positive rate for AUC integration.

    Returns:
        AUPRO value (normalized to [0, 1]).
    """
    if gt_masks.shape != pred_maps.shape:
        raise ValueError(f"Shape mismatch: gt_masks {gt_masks.shape} vs pred_maps {pred_maps.shape}")

    # Step 1: Connected Component Analysis
    cca = np.zeros_like(gt_masks, dtype=np.int32)
    for i in range(len(gt_masks)):
        if gt_masks[i].max() > 0:
            labeled, _ = connected_components(gt_masks[i])
            cca[i] = labeled

    # Make labels unique across batch
    cca = _make_global_region_labels(cca)

    # Step 2: Flatten
    labels = cca.reshape(-1)
    preds_flat = pred_maps.reshape(-1).astype(np.float64)

    # Step 3: Compute contributions
    background = (labels == 0)
    fp_change = background.astype(np.float64)
    num_bg = fp_change.sum()

    if num_bg == 0:
        return 0.0

    max_label = int(labels.max())
    if max_label == 0:
        return 0.0

    # Step 4: Compute region sizes and PRO contribution per pixel
    region_sizes = np.bincount(labels, minlength=max_label + 1).astype(np.float64)
    num_regions = (region_sizes[1:] > 0).sum()

    if num_regions == 0:
        return 0.0

    fg_mask = labels > 0
    pro_change = np.zeros_like(preds_flat)
    pro_change[fg_mask] = 1.0 / region_sizes[labels[fg_mask]]

    # Step 5: Global sort (descending by prediction score)
    idx = np.argsort(preds_flat)[::-1]
    fp_sorted = fp_change[idx]
    pro_sorted = pro_change[idx]
    preds_sorted = preds_flat[idx]

    # Step 6: Cumulative sums
    fpr = np.cumsum(fp_sorted) / num_bg
    pro = np.cumsum(pro_sorted) / num_regions

    fpr = np.clip(fpr, 0.0, 1.0)
    pro = np.clip(pro, 0.0, 1.0)

    # Step 7: Remove duplicate thresholds (keep only unique prediction values)
    keep = np.ones(len(preds_sorted), dtype=bool)
    keep[:-1] = preds_sorted[:-1] != preds_sorted[1:]
    fpr = fpr[keep]
    pro = pro[keep]

    # Prepend zero point
    fpr = np.concatenate([[0.0], fpr])
    pro = np.concatenate([[0.0], pro])

    # Step 8: FPR limit clipping with linear interpolation
    mask = fpr <= max_fpr
    if mask.any():
        i = np.where(mask)[0][-1]
        if fpr[i] < max_fpr and i + 1 < len(fpr):
            f1, f2 = fpr[i], fpr[i + 1]
            p1, p2 = pro[i], pro[i + 1]
            # Linear interpolation
            p_lim = p1 + (p2 - p1) * (max_fpr - f1) / (f2 - f1)
            fpr = np.concatenate([fpr[:i + 1], [max_fpr]])
            pro = np.concatenate([pro[:i + 1], [p_lim]])
        else:
            fpr = fpr[:i + 1]
            pro = pro[:i + 1]
    else:
        return 0.0

    # Step 9: Compute AUC and normalize
    if len(fpr) < 2:
        return 0.0

    aupro = auc(fpr, pro) / max_fpr
    return float(np.clip(aupro, 0.0, 1.0))

"""FPR@TPR metric: False Positive Rate at a given True Positive Rate."""

import numpy as np
from sklearn.metrics import roc_curve


def compute_fpr_at_tpr(
    gt_labels: np.ndarray,
    pred_scores: np.ndarray,
    target_tpr: float = 0.95,
) -> float:
    """Compute FPR at a target TPR (e.g., FPR@95TPR).

    Args:
        gt_labels: Binary ground truth.
        pred_scores: Predicted anomaly scores.
        target_tpr: Target true positive rate.

    Returns:
        FPR value at the given TPR (lower is better).
    """
    if len(np.unique(gt_labels)) < 2:
        return 0.0

    fpr, tpr, _ = roc_curve(gt_labels, pred_scores)

    # Find FPR where TPR >= target_tpr
    idx = np.where(tpr >= target_tpr)[0]
    if len(idx) == 0:
        return 1.0

    return float(fpr[idx[0]])

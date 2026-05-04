"""Official AA-CLIP metric wrapper."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from baoiad.evaluation.ad_metric import AnomalyDetectionMetric
from baoiad.evaluation.aupimo import compute_pimo
from baoiad.evaluation.aupro import compute_aupro
from baoiad.evaluation.ece import compute_ece, compute_pixel_ece
from baoiad.evaluation.fpr_at_tpr import compute_fpr_at_tpr
from baoiad.registry import METRICS


@METRICS.register_module(force=True)
class AACLIPOfficialMetric(AnomalyDetectionMetric):
    """AA-CLIP metric with official per-class normalization and image fusion."""

    def __init__(
        self,
        normalize_pixel_scores: bool = True,
        fuse_image_scores: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.normalize_pixel_scores = bool(normalize_pixel_scores)
        self.fuse_image_scores = bool(fuse_image_scores)

    @staticmethod
    def _official_normalize(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return values
        max_value = float(values.max())
        min_value = float(values.min())
        if np.isclose(max_value, 1.0) or np.isclose(max_value, min_value):
            return values
        return (values - min_value) / (max_value - min_value)

    def _compute_single_class(self, samples: List[Dict]) -> Dict[str, float]:
        gt_labels = np.array([s['gt_label'] for s in samples])
        raw_scores = np.array([s['pred_score'] for s in samples], dtype=np.float64)

        spatial_shape = samples[0].get('gt_mask_shape')
        if spatial_shape is None:
            n_pixels = len(samples[0]['gt_mask'])
            side = int(np.sqrt(n_pixels))
            spatial_shape = (side, side)

        h, w = spatial_shape
        gt_masks_2d = np.array([s['gt_mask'].reshape(h, w) for s in samples])
        pred_maps_2d = np.array([s['pred_anomaly_map'].reshape(h, w) for s in samples])
        gt_masks_flat = gt_masks_2d.reshape(-1)
        gt_masks_flat_binary = gt_masks_flat.astype(np.uint8)
        gt_masks_2d_binary = gt_masks_2d.astype(np.uint8)

        if self.normalize_pixel_scores:
            pred_maps_2d = self._official_normalize(pred_maps_2d)
        pred_maps_flat = pred_maps_2d.reshape(-1)

        pred_scores = self._official_normalize(raw_scores)
        if self.fuse_image_scores:
            pred_scores = 0.5 * pred_maps_2d.max(axis=(1, 2)) + 0.5 * pred_scores

        result: Dict[str, float] = {}
        for metric_name in self.metric_names:
            try:
                if metric_name == 'image_auroc':
                    result[metric_name] = self._safe_auroc(gt_labels, pred_scores)
                elif metric_name == 'pixel_auroc':
                    result[metric_name] = self._safe_auroc(gt_masks_flat_binary, pred_maps_flat)
                elif metric_name == 'image_f1max':
                    result[metric_name] = self._f1_max(gt_labels, pred_scores)
                elif metric_name == 'pixel_f1max':
                    result[metric_name] = self._f1_max(gt_masks_flat_binary, pred_maps_flat)
                elif metric_name == 'image_ap':
                    result[metric_name] = self._safe_ap(gt_labels, pred_scores)
                elif metric_name == 'pixel_ap':
                    result[metric_name] = self._safe_ap(gt_masks_flat_binary, pred_maps_flat)
                elif metric_name == 'aupro':
                    result[metric_name] = compute_aupro(gt_masks_2d, pred_maps_2d)
                elif metric_name == 'aupimo':
                    result[metric_name] = compute_pimo(gt_masks_2d, pred_maps_2d)
                elif metric_name == 'image_ece':
                    result[metric_name] = compute_ece(gt_labels, pred_scores)
                elif metric_name == 'pixel_ece':
                    result[metric_name] = compute_pixel_ece(gt_masks_2d_binary, pred_maps_2d)
                elif metric_name == 'image_fpr@95tpr':
                    result[metric_name] = compute_fpr_at_tpr(gt_labels, pred_scores, 0.95)
            except (ValueError, RuntimeError, IndexError):
                result[metric_name] = 0.0
        return result

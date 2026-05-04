"""Anomaly detection evaluation metrics."""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

from baoiad.evaluation.aupimo import compute_pimo
from baoiad.evaluation.aupro import compute_aupro
from baoiad.evaluation.ece import compute_ece, compute_pixel_ece
from baoiad.evaluation.fpr_at_tpr import compute_fpr_at_tpr
from baoiad.registry import METRICS


@METRICS.register_module(force=True)
class AnomalyDetectionMetric(BaseMetric):
    """Unified anomaly detection metric.

    Computes image-level and pixel-level AUROC, F1-max, and AP.
    Results are grouped by cls_name and averaged.

    Args:
        metrics: List of metric names to compute. Default: all supported.
    """

    SUPPORTED_METRICS = (
        'image_auroc', 'pixel_auroc',
        'image_auroc_mean', 'image_auroc_max',
        'image_f1max', 'pixel_f1max',
        'image_ap', 'pixel_ap',
        'aupro', 'aupimo',
        'image_ece', 'pixel_ece',
        'image_fpr@95tpr',
    )

    default_prefix = 'ad'

    def __init__(
        self,
        metrics: Optional[Sequence[str]] = None,
        resize_mask: Optional[int | Sequence[int]] = None,
        resize_gt_mask_mode: str = 'nearest',
        resize_gt_mask_threshold: Optional[float] = None,
        normalize_image_scores: bool = False,
        normalize_pred_maps: bool | str = False,
        flip_auroc_if_below_half: bool = False,
        image_score_field: str = 'pred_score',
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.metric_names = list(metrics or self.SUPPORTED_METRICS)
        invalid_metrics = [metric for metric in self.metric_names if metric not in self.SUPPORTED_METRICS]
        if invalid_metrics:
            raise ValueError(
                f'Unsupported metrics {invalid_metrics}. '
                f'Supported metrics are {list(self.SUPPORTED_METRICS)}.'
            )
        self.resize_mask = resize_mask
        if resize_gt_mask_mode not in ('nearest', 'bilinear'):
            raise ValueError("resize_gt_mask_mode must be 'nearest' or 'bilinear'.")
        self.resize_gt_mask_mode = resize_gt_mask_mode
        self.resize_gt_mask_threshold = (
            None if resize_gt_mask_threshold is None else float(resize_gt_mask_threshold)
        )
        self.normalize_image_scores = normalize_image_scores
        if normalize_pred_maps is True:
            normalize_pred_maps = 'per_image'
        elif normalize_pred_maps is False:
            normalize_pred_maps = None
        if normalize_pred_maps not in (None, 'per_image', 'batch_broadcast'):
            raise ValueError(
                "normalize_pred_maps must be one of None, 'per_image', or 'batch_broadcast'."
            )
        self.normalize_pred_maps = normalize_pred_maps
        self.flip_auroc_if_below_half = bool(flip_auroc_if_below_half)
        if image_score_field not in ('pred_score', 'pred_score_mean', 'pred_score_max'):
            raise ValueError(
                "image_score_field must be one of 'pred_score', 'pred_score_mean', or 'pred_score_max'."
            )
        self.image_score_field = image_score_field

    def process(self, data_batch: dict, data_samples: Sequence) -> None:
        """Collect predictions from each batch."""
        for sample in data_samples:
            # Support both BaseDataElement and dict
            _is_dict = isinstance(sample, dict)

            def _get(s, k):
                return s[k] if _is_dict else getattr(s, k)

            def _has(s, k):
                return k in s if _is_dict else hasattr(s, k)

            pred_score = float(_get(sample, 'pred_score'))
            result = dict(
                pred_score=pred_score,
                pred_score_mean=float(_get(sample, 'pred_score_mean')) if _has(sample, 'pred_score_mean') else pred_score,
                pred_score_max=float(_get(sample, 'pred_score_max')) if _has(sample, 'pred_score_max') else pred_score,
                gt_label=int(_get(sample, 'gt_label')),
                cls_name=str(_get(sample, 'cls_name') if _has(sample, 'cls_name') else ''),
            )

            gt_mask = _get(sample, 'gt_mask') if _has(sample, 'gt_mask') else np.zeros(1)
            if hasattr(gt_mask, 'cpu'):
                gt_mask = gt_mask.cpu()
            if hasattr(gt_mask, 'numpy'):
                gt_mask = gt_mask.numpy()
            pred_map = _get(sample, 'pred_anomaly_map') if _has(sample, 'pred_anomaly_map') else np.zeros(1)
            if hasattr(pred_map, 'cpu'):
                pred_map = pred_map.cpu()
            if hasattr(pred_map, 'numpy'):
                pred_map = pred_map.numpy()

            if self.resize_mask is not None and gt_mask.size > 1 and pred_map.size > 1:
                import torch
                import torch.nn.functional as Fnn

                if isinstance(self.resize_mask, Sequence):
                    target_h, target_w = int(self.resize_mask[0]), int(self.resize_mask[1])
                else:
                    target_h = target_w = int(self.resize_mask)

                pm = torch.from_numpy(pred_map).float()
                gm = torch.from_numpy(gt_mask).float()

                if pm.ndim == 2:
                    pm = pm.unsqueeze(0).unsqueeze(0)
                elif pm.ndim == 3:
                    pm = pm.unsqueeze(0)
                elif pm.ndim == 1:
                    side = int(pm.shape[0] ** 0.5)
                    pm = pm.view(1, 1, side, side)

                if gm.ndim == 2:
                    gm = gm.unsqueeze(0).unsqueeze(0)
                elif gm.ndim == 3:
                    gm = gm.unsqueeze(0)
                elif gm.ndim == 1:
                    side = int(gm.shape[0] ** 0.5)
                    gm = gm.view(1, 1, side, side)

                pm = Fnn.interpolate(pm, size=(target_h, target_w), mode='bilinear', align_corners=False)
                if self.resize_gt_mask_mode == 'bilinear':
                    gm = Fnn.interpolate(gm, size=(target_h, target_w), mode='bilinear', align_corners=False)
                else:
                    gm = Fnn.interpolate(gm, size=(target_h, target_w), mode='nearest')
                if self.resize_gt_mask_threshold is not None:
                    gm = (gm > self.resize_gt_mask_threshold).float()

                pred_map = pm.squeeze().numpy()
                gt_mask = gm.squeeze().numpy()

            # Align pred_anomaly_map to gt_mask size
            if gt_mask.size > 1 and pred_map.size > 1 and gt_mask.shape != pred_map.shape:
                import torch
                import torch.nn.functional as Fnn
                # pred_map may be (1,H,W) or (H,W), gt_mask may be (H,W)
                pm = torch.from_numpy(pred_map).float()
                if pm.ndim == 1:
                    # Guess it was flattened from square
                    s = int(pm.shape[0] ** 0.5)
                    pm = pm.view(1, 1, s, s)
                elif pm.ndim == 2:
                    pm = pm.unsqueeze(0).unsqueeze(0)
                elif pm.ndim == 3:
                    pm = pm.unsqueeze(0)
                gm = torch.from_numpy(gt_mask).float()
                if gm.ndim == 1:
                    s = int(gm.shape[0] ** 0.5)
                    target_h, target_w = s, s
                else:
                    target_h, target_w = gm.shape[-2], gm.shape[-1]
                pm = Fnn.interpolate(pm, size=(target_h, target_w), mode='bilinear', align_corners=False)
                pred_map = pm.squeeze().numpy()
                gt_mask = gm.numpy() if gm.ndim > 1 else gt_mask

            result.update(dict(
                pred_anomaly_map=pred_map.flatten(),
                gt_mask=gt_mask.flatten(),
                gt_mask_shape=tuple(gt_mask.shape[-2:]) if gt_mask.ndim >= 2 else None,
            ))
            self.results.append(result)

    def compute_metrics(self, results: List[Dict]) -> Dict:
        """Compute all requested metrics, per-class and averaged."""
        grouped: Dict[str, List[Dict]] = defaultdict(list)
        for r in results:
            grouped[r['cls_name']].append(r)

        per_class: Dict[str, Dict[str, float]] = {}
        for cls_name, samples in grouped.items():
            per_class[cls_name] = self._compute_single_class(samples)

        # Average across classes
        out: Dict[str, float] = {}
        for m in self.metric_names:
            values = [per_class[c][m] for c in per_class if m in per_class[c]]
            out[m] = float(np.mean(values)) if values else 0.0

        # Also store per-class results
        for cls_name, cls_metrics in per_class.items():
            for m, v in cls_metrics.items():
                out[f'{cls_name}/{m}'] = v

        return out

    def _compute_single_class(self, samples: List[Dict]) -> Dict[str, float]:
        """Compute metrics for one class."""
        gt_labels = np.array([s['gt_label'] for s in samples])
        pred_scores_by_field = {
            'pred_score': np.array([s['pred_score'] for s in samples], dtype=np.float64),
            'pred_score_mean': np.array([s['pred_score_mean'] for s in samples], dtype=np.float64),
            'pred_score_max': np.array([s['pred_score_max'] for s in samples], dtype=np.float64),
        }
        if self.normalize_image_scores:
            pred_scores_by_field = {
                key: self._minmax_normalize(value)
                for key, value in pred_scores_by_field.items()
            }
        image_scores = pred_scores_by_field[self.image_score_field]

        # For pixel-level: need per-sample 2D arrays for AUPRO/AUPIMO.
        gt_masks_2d = []
        pred_maps_2d = []
        for sample in samples:
            spatial_shape = sample.get('gt_mask_shape')
            if spatial_shape is None:
                n_pixels = len(sample['gt_mask'])
                side = int(np.sqrt(n_pixels))
                spatial_shape = (side, side)
            h, w = spatial_shape
            gt_masks_2d.append(np.asarray(sample['gt_mask']).reshape(h, w))
            pred_maps_2d.append(np.asarray(sample['pred_anomaly_map']).reshape(h, w))

        gt_masks_flat = np.concatenate([s['gt_mask'] for s in samples])
        # Some reference pipelines resize grayscale masks bilinearly, which
        # produces fractional edge pixels. Official metric code bins those
        # masks back to integer labels before AUROC/AP-style pixel metrics.
        gt_masks_flat_binary = gt_masks_flat.astype(np.uint8)

        # 2D versions for region-based metrics
        gt_masks_2d_binary = [gt_mask.astype(np.uint8) for gt_mask in gt_masks_2d]
        if self.normalize_pred_maps is not None:
            pred_maps_2d = self._normalize_pred_maps_2d(pred_maps_2d, self.normalize_pred_maps)
        pred_maps_flat = np.concatenate([pred_map.reshape(-1) for pred_map in pred_maps_2d])

        result: Dict[str, float] = {}

        for m in self.metric_names:
            try:
                if m == 'image_auroc':
                    result[m] = self._safe_auroc(gt_labels, image_scores)
                elif m == 'image_auroc_mean':
                    result[m] = self._safe_auroc(gt_labels, pred_scores_by_field['pred_score_mean'])
                elif m == 'image_auroc_max':
                    result[m] = self._safe_auroc(gt_labels, pred_scores_by_field['pred_score_max'])
                elif m == 'pixel_auroc':
                    result[m] = self._safe_auroc(gt_masks_flat_binary, pred_maps_flat)
                elif m == 'image_f1max':
                    result[m] = self._f1_max(gt_labels, image_scores)
                elif m == 'pixel_f1max':
                    result[m] = self._f1_max(gt_masks_flat_binary, pred_maps_flat)
                elif m == 'image_ap':
                    result[m] = self._safe_ap(gt_labels, image_scores)
                elif m == 'pixel_ap':
                    result[m] = self._safe_ap(gt_masks_flat_binary, pred_maps_flat)
                elif m == 'aupro':
                    result[m] = compute_aupro(gt_masks_2d, pred_maps_2d)
                elif m == 'aupimo':
                    result[m] = compute_pimo(gt_masks_2d, pred_maps_2d)
                elif m == 'image_ece':
                    result[m] = compute_ece(gt_labels, image_scores)
                elif m == 'pixel_ece':
                    result[m] = compute_pixel_ece(gt_masks_2d_binary, pred_maps_2d)
                elif m == 'image_fpr@95tpr':
                    result[m] = compute_fpr_at_tpr(gt_labels, image_scores, 0.95)
            except (ValueError, RuntimeError, IndexError) as e:
                logging.getLogger(__name__).warning(
                    'Failed to compute metric %s: %s', m, e)
                result[m] = 0.0

        return result

    @staticmethod
    def _minmax_normalize(y_score: np.ndarray) -> np.ndarray:
        if y_score.size == 0:
            return y_score
        min_score = float(np.min(y_score))
        max_score = float(np.max(y_score))
        if np.isclose(max_score, min_score):
            return np.zeros_like(y_score, dtype=np.float64)
        return (y_score - min_score) / (max_score - min_score)

    @staticmethod
    def _normalize_pred_maps_2d(pred_maps: Sequence[np.ndarray], mode: str) -> List[np.ndarray]:
        pred_maps = [np.asarray(pred_map, dtype=np.float64).copy() for pred_map in pred_maps]
        mins = np.array([pred_map.reshape(-1).min() for pred_map in pred_maps], dtype=np.float64)
        maxs = np.array([pred_map.reshape(-1).max() for pred_map in pred_maps], dtype=np.float64)

        if mode == 'per_image':
            normalized = []
            for index, (min_score, max_score) in enumerate(zip(mins, maxs)):
                denom = max(float(max_score - min_score), 1e-2)
                normalized.append((pred_maps[index] - min_score) / denom)
            return normalized

        if mode == 'batch_broadcast':
            normalized = [np.zeros_like(pred_map, dtype=np.float64) for pred_map in pred_maps]
            for min_score, max_score in zip(mins, maxs):
                denom = max(float(max_score - min_score), 1e-2)
                for index, pred_map in enumerate(pred_maps):
                    normalized[index] += (pred_map - min_score) / denom
            scale = max(len(pred_maps), 1)
            return [pred_map / scale for pred_map in normalized]

        raise ValueError(f'Unsupported pred map normalization mode: {mode}')

    def _safe_auroc(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        if len(np.unique(y_true)) < 2:
            return 0.0
        auc = float(roc_auc_score(y_true, y_score))
        if self.flip_auroc_if_below_half and auc < 0.5:
            return 1.0 - auc
        return auc

    @staticmethod
    def _safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
        if len(np.unique(y_true)) < 2:
            return 0.0
        return float(average_precision_score(y_true, y_score))

    @staticmethod
    def _f1_max(y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Compute maximum F1 score using precision-recall curve."""
        if len(np.unique(y_true)) < 2:
            return 0.0
        # Ensure y_true is binary for precision_recall_curve
        y_true_bin = (y_true > 0.5).astype(int)
        precision, recall, _ = precision_recall_curve(y_true_bin, y_score)
        # F1 = 2 * P * R / (P + R), handle division by zero
        denom = precision + recall
        f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)
        return float(np.max(f1))

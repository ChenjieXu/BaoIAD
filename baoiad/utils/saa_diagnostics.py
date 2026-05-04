"""Helpers for targeted SAA score diagnostics.

These utilities are intentionally lightweight and read-only relative to the
main detector logic. They let diagnosis scripts compare raw per-image scores,
post-normalization scores, and label-wise score overlap without changing the
main evaluation path.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from baoiad.models.detectors.saa_prompts import parse_property_prompt


def resolve_saa_object_controls(
    cls_name: str,
    property_prompt: str | None,
    *,
    default_k_mask: int,
    default_defect_area_threshold: float,
) -> Dict[str, Any]:
    """Resolve runtime object controls from an optional property prompt."""
    controls = {
        'property_prompt': property_prompt,
        'object_prompt': cls_name,
        'object_number': 1,
        'object_max_area': 1.0,
        'k_mask': int(default_k_mask),
        'defect_area_threshold': float(default_defect_area_threshold),
    }
    if property_prompt is None:
        return controls

    props = parse_property_prompt(property_prompt)
    controls.update({
        'object_prompt': props.get('object_prompt', cls_name),
        'object_number': int(props.get('object_number', 1)),
        'object_max_area': float(props.get('object_max_area', 1.0)),
        'k_mask': int(props.get('k_mask', default_k_mask)),
        'defect_area_threshold': float(
            props.get('defect_area_threshold', default_defect_area_threshold)
        ),
    })
    return controls


def summarize_saa_object_path(
    object_masks: torch.Tensor | np.ndarray | None,
    *,
    object_number: int,
) -> Dict[str, Any]:
    """Summarize whether SAA saliency resolves to single- or multi-instance."""
    if object_masks is None:
        object_mask_count = 0
        non_empty_count = 0
    elif torch.is_tensor(object_masks):
        masks = object_masks.detach().float().cpu()
        object_mask_count = int(masks.shape[0]) if masks.ndim >= 1 else 0
        if object_mask_count == 0 or masks.ndim < 3:
            non_empty_count = 0
        else:
            non_empty_count = int((masks.reshape(object_mask_count, -1).sum(dim=1) > 0).sum().item())
    else:
        masks = np.asarray(object_masks, dtype=np.float32)
        object_mask_count = int(masks.shape[0]) if masks.ndim >= 1 else 0
        if object_mask_count == 0 or masks.ndim < 3:
            non_empty_count = 0
        else:
            non_empty_count = int((masks.reshape(object_mask_count, -1).sum(axis=1) > 0).sum())

    if object_number <= 1:
        saliency_strategy = 'single'
    elif non_empty_count <= 1:
        saliency_strategy = 'single_fallback'
    else:
        saliency_strategy = 'multi'

    return {
        'object_number': int(object_number),
        'object_mask_count': int(object_mask_count),
        'object_mask_non_empty_count': int(non_empty_count),
        'saliency_strategy': saliency_strategy,
    }


def normalize_score_maps(score_maps: torch.Tensor) -> torch.Tensor:
    """Apply the same dataset-level min-max normalization as ``score_all()``."""
    maps = score_maps.detach().float().cpu()
    min_value = maps.min()
    max_value = maps.max()
    if float(max_value.item() - min_value.item()) > 1e-12:
        return (maps - min_value) / (max_value - min_value)
    return torch.zeros_like(maps)


def minmax_normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    """Apply dataset-level min-max normalization to image-level scores."""
    values = scores.detach().float().cpu()
    min_value = values.min()
    max_value = values.max()
    if float(max_value.item() - min_value.item()) > 1e-12:
        return (values - min_value) / (max_value - min_value)
    return torch.zeros_like(values)


def aggregate_image_score(
    anomaly_map: torch.Tensor,
    mode: str = 'map_max',
    topk_scores: torch.Tensor | None = None,
) -> float:
    """Aggregate one anomaly map into an image-level score.

    Args:
        anomaly_map: Tensor with shape ``(1, H, W)`` or ``(H, W)``.
        mode: Aggregation mode. Supported:
            - ``map_max``: max over the anomaly map.
            - ``map_p99``: 99th percentile over the anomaly map.
            - ``support_mean``: mean over non-zero support in the anomaly map.
            - ``topk_combined_score_max``: max over selected proposal scores.
            - ``topk_combined_score_mean``: mean over the selected top-k
              proposal scores used for image ranking.
        topk_scores: Selected proposal scores for score-only aggregation modes.
    """
    amap = anomaly_map.detach().float().cpu()
    flat = amap.reshape(-1)

    if mode == 'map_max':
        return float(flat.max().item())
    if mode == 'map_p99':
        return float(torch.quantile(flat, 0.99).item())
    if mode == 'support_mean':
        support = flat > 0
        if bool(support.any().item()):
            return float(flat[support].mean().item())
        return 0.0
    if mode == 'topk_combined_score_max':
        if topk_scores is None or len(topk_scores) == 0:
            return 0.0
        return float(topk_scores.detach().float().cpu().max().item())
    if mode == 'topk_combined_score_mean':
        if topk_scores is None or len(topk_scores) == 0:
            return 0.0
        return float(topk_scores.detach().float().cpu().mean().item())
    raise ValueError(f'Unsupported aggregation mode: {mode}')


def image_scores_from_maps(
    score_maps: torch.Tensor,
    mode: str = 'map_max',
) -> torch.Tensor:
    """Compute image-level scores from anomaly maps."""
    maps = score_maps.detach().float().cpu()
    return torch.tensor(
        [aggregate_image_score(anomaly_map, mode=mode) for anomaly_map in maps],
        dtype=torch.float32,
    )


def select_topk_indices(rank_scores: torch.Tensor, k_mask: int) -> torch.Tensor:
    """Select top-k indices based on a ranking tensor."""
    scores = rank_scores.detach().float().cpu()
    if len(scores) <= k_mask:
        return torch.arange(len(scores), dtype=torch.long)
    return scores.topk(k_mask).indices.detach().cpu()


def labelwise_score_stats(labels: Sequence[int], scores: Sequence[float]) -> Dict[str, float | None]:
    """Summarize score overlap for normal vs anomalous images."""
    label_array = np.asarray(labels, dtype=np.int64)
    score_array = np.asarray(scores, dtype=np.float64)
    normal_scores = score_array[label_array == 0]
    anomaly_scores = score_array[label_array == 1]

    def _maybe(name: str, values: np.ndarray, fn) -> Dict[str, float | None]:
        if values.size == 0:
            return {name: None}
        return {name: float(fn(values))}

    summary: Dict[str, float | None] = {}
    summary.update(_maybe('normal_mean', normal_scores, np.mean))
    summary.update(_maybe('normal_min', normal_scores, np.min))
    summary.update(_maybe('normal_max', normal_scores, np.max))
    summary.update(_maybe('anomaly_mean', anomaly_scores, np.mean))
    summary.update(_maybe('anomaly_min', anomaly_scores, np.min))
    summary.update(_maybe('anomaly_max', anomaly_scores, np.max))
    if normal_scores.size and anomaly_scores.size:
        summary['gap_mean'] = float(np.mean(anomaly_scores) - np.mean(normal_scores))
        summary['gap_boundary'] = float(np.min(anomaly_scores) - np.max(normal_scores))
    else:
        summary['gap_mean'] = None
        summary['gap_boundary'] = None
    return summary


def binary_ranking_metrics(labels: Sequence[int], scores: Sequence[float]) -> Dict[str, float | None]:
    """Compute image-level ranking metrics when both classes are present."""
    label_array = np.asarray(labels, dtype=np.int64)
    score_array = np.asarray(scores, dtype=np.float64)
    metrics: Dict[str, float | None] = {
        'image_auroc': None,
        'image_ap': None,
    }
    if np.unique(label_array).size < 2:
        return metrics

    metrics['image_auroc'] = float(roc_auc_score(label_array, score_array))
    metrics['image_ap'] = float(average_precision_score(label_array, score_array))
    return metrics


def build_image_score_summary(
    labels: Sequence[int],
    raw_scores: Sequence[float],
    normalized_scores: Sequence[float],
) -> Dict[str, Any]:
    """Summarize raw and normalized image-level scores."""
    return {
        'num_samples': int(len(labels)),
        'raw': {
            **binary_ranking_metrics(labels, raw_scores),
            **labelwise_score_stats(labels, raw_scores),
        },
        'normalized': {
            **binary_ranking_metrics(labels, normalized_scores),
            **labelwise_score_stats(labels, normalized_scores),
        },
    }


def records_to_csv_rows(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten diagnostic records for CSV export."""
    rows: List[Dict[str, Any]] = []
    for record in records:
        rows.append({
            'index': record['index'],
            'img_path': record['img_path'],
            'defect_type': record['defect_type'],
            'gt_label': record['gt_label'],
            'raw_image_score': record['raw_image_score'],
            'normalized_image_score': record.get('normalized_image_score'),
            'raw_map_max': record['raw_map_max'],
            'raw_map_mean': record['raw_map_mean'],
            'raw_map_p99': record['raw_map_p99'],
            'object_area': record['object_area'],
            'defect_max_area': record['defect_max_area'],
            'object_number': record.get('object_number'),
            'object_mask_count': record.get('object_mask_count'),
            'object_mask_non_empty_count': record.get('object_mask_non_empty_count'),
            'saliency_strategy': record.get('saliency_strategy'),
            'num_prompts': record['num_prompts'],
            'num_boxes': record['num_boxes'],
            'det_score_max': record['det_score_max'],
            'det_score_mean': record['det_score_mean'],
            'saliency_score_max': record['saliency_score_max'],
            'saliency_score_mean': record['saliency_score_mean'],
            'combined_score_max': record['combined_score_max'],
            'combined_score_mean': record['combined_score_mean'],
            'aggregation_mode': record['aggregation_mode'],
            'topk_rank_mode': record['topk_rank_mode'],
            'image_score_rank_mode': record.get('image_score_rank_mode'),
            'saliency_score_mode': record.get('saliency_score_mode'),
            'saliency_score_clip_max': record.get('saliency_score_clip_max'),
            'image_score_area_range': record.get('image_score_area_range'),
            'selected_image_score_count': record.get('selected_image_score_count'),
            'selected_image_score_area_ratio_max': record.get('selected_image_score_area_ratio_max'),
            'selected_image_score_area_ratio_mean': record.get('selected_image_score_area_ratio_mean'),
            'selected_rank_score_max': record['selected_rank_score_max'],
            'selected_rank_score_mean': record['selected_rank_score_mean'],
            'top_phrases': '|'.join(record['top_phrases']),
            'image_score_phrases': '|'.join(record.get('image_score_phrases', [])),
        })
    return rows

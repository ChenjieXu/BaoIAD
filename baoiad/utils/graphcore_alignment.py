"""Shared GraphCore alignment helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

GRAPHCORE_CLASSES = (
    'bottle',
    'cable',
    'capsule',
    'carpet',
    'grid',
    'hazelnut',
    'leather',
    'metal_nut',
    'pill',
    'screw',
    'tile',
    'toothbrush',
    'transistor',
    'wood',
    'zipper',
)
GRAPHCORE_IMAGE_SCORE_MODES = (
    'raw_max',
    'raw_mean',
    'raw_p95',
    'raw_p99',
    'smooth_max',
    'smooth_mean',
    'smooth_p95',
    'smooth_p99',
)
GRAPHCORE_STRICT_IMAGE_SCORE_MODE = 'raw_max'
GRAPHCORE_STRICT_CORESET_INITIAL_INDEX = 0
GRAPHCORE_ORDER_DIR = 'runs/alignment/graphcore_orders'
GRAPHCORE_COMPARE_REPORT_GLOB = 'graphcore_official_compare*.json'
GRAPHCORE_COMPARE_PIXEL_METRICS = ('pixel_auroc', 'aupro')


def normalize_graphcore_image_score_mode(mode: str) -> str:
    normalized = str(mode)
    if normalized not in GRAPHCORE_IMAGE_SCORE_MODES:
        raise ValueError(f'Unsupported GraphCore image score mode: {mode}')
    return normalized


def normalize_graphcore_image_score_mode_overrides(
    overrides: Mapping[str, str] | None,
) -> dict[str, str]:
    normalized: dict[str, str] = {}
    if overrides is None:
        return normalized
    for cls_name, mode in dict(overrides).items():
        normalized[str(cls_name)] = normalize_graphcore_image_score_mode(mode)
    return normalized


def reduce_graphcore_image_score(
    patch_map: np.ndarray,
    smooth_map: np.ndarray,
    mode: str,
) -> float:
    patch_map = np.asarray(patch_map, dtype=np.float64)
    smooth_map = np.asarray(smooth_map, dtype=np.float64)
    normalized_mode = normalize_graphcore_image_score_mode(mode)

    if normalized_mode == 'raw_max':
        return float(patch_map.max())
    if normalized_mode == 'raw_mean':
        return float(patch_map.mean())
    if normalized_mode == 'raw_p95':
        return float(np.percentile(patch_map, 95))
    if normalized_mode == 'raw_p99':
        return float(np.percentile(patch_map, 99))
    if normalized_mode == 'smooth_max':
        return float(smooth_map.max())
    if normalized_mode == 'smooth_mean':
        return float(smooth_map.mean())
    if normalized_mode == 'smooth_p95':
        return float(np.percentile(smooth_map, 95))
    return float(np.percentile(smooth_map, 99))


def graphcore_strict_alignment_violations(
    model_cfg: Mapping[str, Any] | None,
) -> list[str]:
    cfg = dict(model_cfg or {})
    violations: list[str] = []

    try:
        mode = normalize_graphcore_image_score_mode(
            cfg.get('image_score_mode', GRAPHCORE_STRICT_IMAGE_SCORE_MODE)
        )
    except ValueError as err:
        violations.append(str(err))
    else:
        if mode != GRAPHCORE_STRICT_IMAGE_SCORE_MODE:
            violations.append(
                'GraphCore strict alignment requires '
                f"image_score_mode='{GRAPHCORE_STRICT_IMAGE_SCORE_MODE}', got '{mode}'."
            )

    try:
        overrides = normalize_graphcore_image_score_mode_overrides(
            cfg.get('image_score_mode_overrides')
        )
    except ValueError as err:
        violations.append(str(err))
    else:
        if overrides:
            violations.append(
                'GraphCore strict alignment forbids diagnose-only '
                f'image_score_mode_overrides, got {sorted(overrides)}.'
            )

    initial_index = cfg.get(
        'coreset_initial_index',
        GRAPHCORE_STRICT_CORESET_INITIAL_INDEX,
    )
    try:
        normalized_initial_index = int(initial_index)
    except (TypeError, ValueError):
        violations.append(
            'GraphCore strict alignment requires an integer '
            f'coreset_initial_index, got {initial_index!r}.'
        )
    else:
        if normalized_initial_index != GRAPHCORE_STRICT_CORESET_INITIAL_INDEX:
            violations.append(
                'GraphCore strict alignment requires '
                f'coreset_initial_index={GRAPHCORE_STRICT_CORESET_INITIAL_INDEX}, '
                f'got {normalized_initial_index}.'
            )

    return violations


def graphcore_order_file(
    class_name: str,
    order_root: str = GRAPHCORE_ORDER_DIR,
) -> Path:
    return Path(order_root) / f'{class_name}.json'


def graphcore_explicit_order_cfg_overrides(
    class_name: str,
    order_root: str = GRAPHCORE_ORDER_DIR,
) -> dict[str, str]:
    order_file = graphcore_order_file(class_name, order_root=order_root)
    if not order_file.exists():
        return {}
    return {
        'train_dataloader.sampler.type': 'ExplicitOrderSampler',
        'train_dataloader.sampler.index_file': str(order_file),
        'train_dataloader.sampler.round_up': False,
    }


def _graphcore_metric_value(
    metrics: Mapping[str, Any] | None,
    key: str,
) -> float | None:
    if not isinstance(metrics, Mapping) or key not in metrics:
        return None
    try:
        value = float(metrics[key])
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def graphcore_compare_report_info(
    report_path: str | Path,
) -> dict[str, Any] | None:
    path = Path(report_path)
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None

    class_name = str(payload.get('class_name', ''))
    if class_name not in GRAPHCORE_CLASSES:
        return None

    official_metrics = {
        'image_auroc': _graphcore_metric_value(payload.get('official'), 'image_auroc'),
        'pixel_auroc': _graphcore_metric_value(payload.get('official'), 'pixel_auroc'),
        'aupro': _graphcore_metric_value(payload.get('official'), 'aupro'),
    }
    baoiad_metrics = {
        'image_auroc': _graphcore_metric_value(payload.get('baoiad'), 'image_auroc'),
        'pixel_auroc': _graphcore_metric_value(payload.get('baoiad'), 'pixel_auroc'),
        'aupro': _graphcore_metric_value(payload.get('baoiad'), 'aupro'),
    }
    return {
        'path': str(path),
        'class_name': class_name,
        'mtime_ns': path.stat().st_mtime_ns,
        'train_order_exact': bool(
            dict(payload.get('train_order_compare') or {}).get('exact_match', False)
        ),
        'official_metrics': official_metrics,
        'baoiad_metrics': baoiad_metrics,
        'image_gap_abs': (
            abs(official_metrics['image_auroc'] - baoiad_metrics['image_auroc'])
            if (
                official_metrics['image_auroc'] is not None
                and baoiad_metrics['image_auroc'] is not None
            )
            else float('inf')
        ),
        'pixel_gap_abs': (
            abs(official_metrics['pixel_auroc'] - baoiad_metrics['pixel_auroc'])
            if (
                official_metrics['pixel_auroc'] is not None
                and baoiad_metrics['pixel_auroc'] is not None
            )
            else float('inf')
        ),
        'has_image_metrics': (
            official_metrics['image_auroc'] is not None
            and baoiad_metrics['image_auroc'] is not None
        ),
        'has_pixel_metrics': all(
            official_metrics[name] is not None and baoiad_metrics[name] is not None
            for name in GRAPHCORE_COMPARE_PIXEL_METRICS
        ),
        'payload': payload,
    }


def graphcore_collect_compare_reports(
    report_root: str | Path = 'runs/alignment',
    class_names: tuple[str, ...] = GRAPHCORE_CLASSES,
) -> dict[str, list[dict[str, Any]]]:
    classes = set(class_names)
    collected: dict[str, list[dict[str, Any]]] = {name: [] for name in class_names}
    for path in sorted(Path(report_root).glob(GRAPHCORE_COMPARE_REPORT_GLOB)):
        info = graphcore_compare_report_info(path)
        if info is None or info['class_name'] not in classes:
            continue
        collected[info['class_name']].append(info)
    return collected


def graphcore_select_compare_reports(
    report_root: str | Path = 'runs/alignment',
    class_names: tuple[str, ...] = GRAPHCORE_CLASSES,
) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for class_name, reports in graphcore_collect_compare_reports(
        report_root=report_root,
        class_names=class_names,
    ).items():
        if not reports:
            continue
        selected[class_name] = max(
            reports,
            key=lambda item: (
                item['has_image_metrics'],
                item['train_order_exact'],
                item['has_pixel_metrics'],
                -item['image_gap_abs'],
                -item['pixel_gap_abs'],
                item['mtime_ns'],
            ),
        )
    return selected


def graphcore_compare_report_summary(
    report_root: str | Path = 'runs/alignment',
    class_names: tuple[str, ...] = GRAPHCORE_CLASSES,
) -> dict[str, Any]:
    selected = graphcore_select_compare_reports(
        report_root=report_root,
        class_names=class_names,
    )
    missing_reports = [name for name in class_names if name not in selected]
    missing_exact_order = [
        name for name, info in selected.items() if not info['train_order_exact']
    ]
    missing_image = [
        name for name, info in selected.items() if not info['has_image_metrics']
    ]
    missing_pixel = [
        name for name, info in selected.items() if not info['has_pixel_metrics']
    ]

    def _mean(metric_name: str, side: str) -> float | None:
        values = []
        for info in selected.values():
            value = info[f'{side}_metrics'][metric_name]
            if value is not None:
                values.append(value)
        if not values:
            return None
        return float(np.mean(values))

    return {
        'class_names': list(class_names),
        'selected_reports': {
            class_name: {
                'path': info['path'],
                'train_order_exact': info['train_order_exact'],
                'has_image_metrics': info['has_image_metrics'],
                'has_pixel_metrics': info['has_pixel_metrics'],
                'official_metrics': info['official_metrics'],
                'baoiad_metrics': info['baoiad_metrics'],
            }
            for class_name, info in selected.items()
        },
        'coverage': {
            'reports_found': len(selected),
            'missing_reports': missing_reports,
            'missing_exact_order': missing_exact_order,
            'missing_image_metrics': missing_image,
            'missing_pixel_metrics': missing_pixel,
            'strict_closure_ready': not (
                missing_reports
                or missing_exact_order
                or missing_image
                or missing_pixel
            ),
        },
        'means': {
            'official': {
                'image_auroc': _mean('image_auroc', 'official'),
                'pixel_auroc': _mean('pixel_auroc', 'official'),
                'aupro': _mean('aupro', 'official'),
            },
            'baoiad': {
                'image_auroc': _mean('image_auroc', 'baoiad'),
                'pixel_auroc': _mean('pixel_auroc', 'baoiad'),
                'aupro': _mean('aupro', 'baoiad'),
            },
        },
    }


def graphcore_strict_closure_violations(
    report_root: str | Path = 'runs/alignment',
    class_names: tuple[str, ...] = GRAPHCORE_CLASSES,
) -> list[str]:
    summary = graphcore_compare_report_summary(
        report_root=report_root,
        class_names=class_names,
    )
    coverage = summary['coverage']
    violations = []
    if coverage['missing_reports']:
        violations.append(
            'GraphCore strict closure is missing official compare reports for '
            f"{', '.join(coverage['missing_reports'])}."
        )
    if coverage['missing_exact_order']:
        violations.append(
            'GraphCore strict closure requires exact-order compare reports for '
            f"{', '.join(coverage['missing_exact_order'])}."
        )
    if coverage['missing_image_metrics']:
        violations.append(
            'GraphCore strict closure is missing image-side metrics in selected compare reports for '
            f"{', '.join(coverage['missing_image_metrics'])}."
        )
    if coverage['missing_pixel_metrics']:
        violations.append(
            'GraphCore strict closure is missing pixel-side metrics '
            '(pixel_auroc + aupro) in selected compare reports for '
            f"{', '.join(coverage['missing_pixel_metrics'])}."
        )
    return violations

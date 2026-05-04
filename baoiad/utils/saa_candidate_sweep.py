"""Helpers for vanilla-SAA candidate recovery sweeps.

These helpers keep experimental pill-candidate exploration separate from the
strict baseline config. They build stable prompt pools, generate deterministic
variant grids, and summarize targeted ``good`` vs defect-type ranking results.
"""

from __future__ import annotations

from itertools import product
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from baoiad.models.detectors.saa_prompts import MVTEC_MANUAL_PROMPTS, build_saa_prompts


SAM_INPUT_MODES = {
    'rgb_sam': True,
    'bgr_sam': False,
}

PROMPT_POOL_ALIASES = {
    'general': 'general_only',
    'manual_old': 'general_plus_manual',
    'manual_area1': 'general_plus_manual',
    'combined_manual_imprint': 'general_plus_manual_plus_imprint',
}

DEFAULT_PROMPT_POOLS = (
    'general_only',
    'manual_only',
    'imprint_only',
    'general_plus_manual',
    'general_plus_manual_plus_imprint',
)
DEFAULT_K_MASKS = (1, 5)
DEFAULT_DEFECT_AREA_THRESHOLDS = (0.5, 1.0)
DEFAULT_SAM_INPUT_MODES = ('rgb_sam', 'bgr_sam')


def _canonical_cls_name(cls_name: str) -> str:
    return cls_name.lower().replace(' ', '_')


def build_prompt_pools(cls_name: str) -> Dict[str, List[tuple[str, str]]]:
    """Build named prompt pools for one SAA class."""
    general_prompts, _ = build_saa_prompts(cls_name, mode='saa')
    manual_prompts = [
        (prompt, filter_phrase)
        for prompt, filter_phrase in MVTEC_MANUAL_PROMPTS.get(_canonical_cls_name(cls_name), [])
    ]
    imprint_prompts = [('imprint', cls_name)]

    pools: Dict[str, List[tuple[str, str]]] = {
        'general_only': list(general_prompts),
        'imprint_only': list(imprint_prompts),
    }
    if manual_prompts:
        pools['manual_only'] = list(manual_prompts)
        pools['general_plus_manual'] = list(general_prompts) + list(manual_prompts)
        pools['general_plus_manual_plus_imprint'] = (
            list(general_prompts) + list(manual_prompts) + list(imprint_prompts)
        )
    return pools


def resolve_prompt_pool_names(
    cls_name: str,
    prompt_pools: Sequence[str] | None = None,
) -> List[str]:
    """Resolve prompt-pool aliases against the class-specific pool inventory."""
    available = build_prompt_pools(cls_name)
    if prompt_pools:
        requested = list(prompt_pools)
    else:
        requested = [name for name in DEFAULT_PROMPT_POOLS if name in available]

    resolved: List[str] = []
    for name in requested:
        canonical = PROMPT_POOL_ALIASES.get(name, name)
        if canonical not in available:
            raise ValueError(
                f'Prompt pool {name!r} is unavailable for class {cls_name!r}. '
                f'Available pools: {sorted(available)}'
            )
        if canonical not in resolved:
            resolved.append(canonical)
    return resolved


def _format_area_threshold(value: float) -> str:
    return str(float(value)).replace('.', 'p')


def build_candidate_variants(
    cls_name: str,
    prompt_pools: Sequence[str] | None = None,
    k_masks: Sequence[int] = DEFAULT_K_MASKS,
    defect_area_thresholds: Sequence[float] = DEFAULT_DEFECT_AREA_THRESHOLDS,
    sam_input_modes: Sequence[str] = DEFAULT_SAM_INPUT_MODES,
) -> List[Dict[str, object]]:
    """Build the vanilla-SAA recovery matrix for one class."""
    resolved_prompt_pools = resolve_prompt_pool_names(cls_name, prompt_pools)
    prompt_pool_map = build_prompt_pools(cls_name)

    variants: List[Dict[str, object]] = []
    for prompt_pool_name, k_mask, area_threshold, sam_input_mode in product(
        resolved_prompt_pools,
        k_masks,
        defect_area_thresholds,
        sam_input_modes,
    ):
        if sam_input_mode not in SAM_INPUT_MODES:
            raise ValueError(
                f'Unsupported sam_input_mode {sam_input_mode!r}. '
                f'Expected one of {sorted(SAM_INPUT_MODES)}'
            )
        variants.append({
            'name': (
                f'{prompt_pool_name}_k{k_mask}_'
                f'area{_format_area_threshold(area_threshold)}_{sam_input_mode}'
            ),
            'cls_name': cls_name,
            'prompt_pool_name': prompt_pool_name,
            'prompts': list(prompt_pool_map[prompt_pool_name]),
            'num_prompts': len(prompt_pool_map[prompt_pool_name]),
            'k_mask': int(k_mask),
            'defect_area_threshold': float(area_threshold),
            'sam_input_mode': sam_input_mode,
            'sam_preconvert_rgb': SAM_INPUT_MODES[sam_input_mode],
        })
    return variants


def summarize_good_vs_anomaly(
    good_scores: Sequence[float],
    anomaly_scores: Sequence[float],
) -> Dict[str, float | None]:
    """Summarize ranking quality for one normal-vs-anomaly split."""
    normal = np.asarray(good_scores, dtype=np.float64)
    anomaly = np.asarray(anomaly_scores, dtype=np.float64)
    if normal.size == 0 or anomaly.size == 0:
        return {
            'image_auroc': None,
            'image_ap': None,
            'normal_mean': None,
            'anomaly_mean': None,
        }

    labels = np.concatenate([
        np.zeros(normal.size, dtype=np.int64),
        np.ones(anomaly.size, dtype=np.int64),
    ])
    scores = np.concatenate([normal, anomaly])
    return {
        'image_auroc': float(roc_auc_score(labels, scores)),
        'image_ap': float(average_precision_score(labels, scores)),
        'normal_mean': float(np.mean(normal)),
        'anomaly_mean': float(np.mean(anomaly)),
    }


def summarize_candidate_scores(
    scores_by_type: Mapping[str, Sequence[float]],
    good_type: str = 'good',
) -> Dict[str, object]:
    """Build one compact candidate-summary payload from per-type scores."""
    good_scores = list(scores_by_type.get(good_type, []))
    if not good_scores:
        raise ValueError('scores_by_type must include non-empty good scores')

    anomaly_scores: List[float] = []
    per_type: Dict[str, Dict[str, float | None]] = {}
    counts: Dict[str, int] = {}
    for defect_type, values in scores_by_type.items():
        counts[defect_type] = int(len(values))
        if defect_type == good_type:
            continue
        defect_scores = list(values)
        anomaly_scores.extend(defect_scores)
        per_type[defect_type] = summarize_good_vs_anomaly(good_scores, defect_scores)

    summary = summarize_good_vs_anomaly(good_scores, anomaly_scores)
    return {
        'num_samples': int(sum(counts.values())),
        'counts': counts,
        'summary': summary,
        'per_type': per_type,
    }

"""Helpers for GraphCore alignment diagnosis."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from baoiad.utils.graphcore_alignment import reduce_graphcore_image_score


GRAPHCORE_ROOT_CAUSES = (
    'preprocess/order drift',
    'feature drift',
    'coreset drift',
    'score-ranking drift',
    'official instability',
)


def safe_auroc(labels: Sequence[int] | np.ndarray, scores: Sequence[float] | np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.size == 0 or np.unique(labels).size < 2:
        return 0.0
    return float(roc_auc_score(labels, scores))


def safe_ap(labels: Sequence[int] | np.ndarray, scores: Sequence[float] | np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.size == 0 or np.unique(labels).size < 2:
        return 0.0
    return float(average_precision_score(labels, scores))


def safe_fpr_at_tpr(
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    target_tpr: float = 0.95,
) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.size == 0 or np.unique(labels).size < 2:
        return 0.0
    fpr, tpr, _ = roc_curve(labels, scores)
    hit = np.where(tpr >= target_tpr)[0]
    if hit.size == 0:
        return 1.0
    return float(fpr[int(hit[0])])


def stats_dict(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            'count': 0.0,
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'p05': 0.0,
            'p50': 0.0,
            'p95': 0.0,
            'p99': 0.0,
        }
    return {
        'count': float(values.size),
        'mean': float(values.mean()),
        'std': float(values.std()),
        'min': float(values.min()),
        'max': float(values.max()),
        'p05': float(np.percentile(values, 5)),
        'p50': float(np.percentile(values, 50)),
        'p95': float(np.percentile(values, 95)),
        'p99': float(np.percentile(values, 99)),
    }


def tensor_stats(tensor: torch.Tensor) -> dict[str, float | list[int] | str]:
    data = tensor.detach().float().cpu()
    return {
        'shape': list(data.shape),
        'dtype': str(data.dtype),
        'mean': float(data.mean().item()),
        'std': float(data.std(unbiased=False).item()),
        'min': float(data.min().item()),
        'max': float(data.max().item()),
    }


def vector_norm_stats(array: np.ndarray) -> dict[str, float]:
    if array.size == 0:
        return stats_dict(np.array([], dtype=np.float64))
    values = np.linalg.norm(array.astype(np.float64), axis=1)
    return stats_dict(values)


def image_score_from_maps(
    patch_map: np.ndarray,
    smooth_map: np.ndarray,
    mode: str,
) -> float:
    return reduce_graphcore_image_score(patch_map, smooth_map, mode)


def summarize_scores(
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
) -> dict[str, object]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]
    return {
        'image_auroc': safe_auroc(labels, scores),
        'image_ap': safe_ap(labels, scores),
        'image_fpr@95tpr': safe_fpr_at_tpr(labels, scores),
        'all': stats_dict(scores),
        'normal': stats_dict(normal_scores),
        'anomaly': stats_dict(anomaly_scores),
        'gap_mean': float(anomaly_scores.mean() - normal_scores.mean()) if normal_scores.size and anomaly_scores.size else 0.0,
    }


def jaccard_index(a: Iterable[int], b: Iterable[int]) -> float:
    a_set = set(int(x) for x in a)
    b_set = set(int(x) for x in b)
    union = a_set | b_set
    if not union:
        return 1.0
    return float(len(a_set & b_set) / len(union))


def best_mode_summary(
    score_summary: Mapping[str, Mapping[str, Any]],
    baseline_mode: str = 'raw_max',
) -> dict[str, Any]:
    if not score_summary:
        return {
            'mode': None,
            'image_auroc': 0.0,
            'baseline_mode': baseline_mode,
            'baseline_image_auroc': 0.0,
            'gain_vs_baseline': 0.0,
            'family': 'unknown',
        }

    baseline_image_auroc = float(score_summary.get(baseline_mode, {}).get('image_auroc', 0.0))
    best_mode, best_payload = max(
        score_summary.items(),
        key=lambda item: float(item[1].get('image_auroc', float('-inf'))),
    )
    best_image_auroc = float(best_payload.get('image_auroc', 0.0))
    return {
        'mode': best_mode,
        'image_auroc': best_image_auroc,
        'baseline_mode': baseline_mode,
        'baseline_image_auroc': baseline_image_auroc,
        'gain_vs_baseline': best_image_auroc - baseline_image_auroc,
        'family': 'smooth' if str(best_mode).startswith('smooth_') else 'raw',
    }


def summarize_feature_preview_deltas(
    feature_previews: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, Any]:
    hook_mean_deltas = []
    hook_std_deltas = []
    embedding_mean_deltas = []
    embedding_std_deltas = []
    num_pairs = 0

    for preview in dict(feature_previews or {}).values():
        official = dict(preview.get('official', {}))
        baoiad = dict(preview.get('baoiad', {}))
        official_hooks = official.get('hook_stats', [])
        baoiad_hooks = baoiad.get('hook_stats', [])
        for off_hook, iad_hook in zip(official_hooks, baoiad_hooks):
            hook_mean_deltas.append(abs(float(iad_hook.get('mean', 0.0)) - float(off_hook.get('mean', 0.0))))
            hook_std_deltas.append(abs(float(iad_hook.get('std', 0.0)) - float(off_hook.get('std', 0.0))))

        off_embedding = official.get('embedding_stats')
        iad_embedding = baoiad.get('embedding_stats')
        if isinstance(off_embedding, Mapping) and isinstance(iad_embedding, Mapping):
            embedding_mean_deltas.append(
                abs(float(iad_embedding.get('mean', 0.0)) - float(off_embedding.get('mean', 0.0)))
            )
            embedding_std_deltas.append(
                abs(float(iad_embedding.get('std', 0.0)) - float(off_embedding.get('std', 0.0)))
            )
            num_pairs += 1

    return {
        'num_preview_pairs': int(num_pairs),
        'hook_mean_abs_diff': stats_dict(hook_mean_deltas),
        'hook_std_abs_diff': stats_dict(hook_std_deltas),
        'embedding_mean_abs_diff': stats_dict(embedding_mean_deltas),
        'embedding_std_abs_diff': stats_dict(embedding_std_deltas),
    }


def classify_graphcore_diagnose_root_cause(
    current_summary: Mapping[str, Mapping[str, Any]],
    official_zero_init_summary: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    current_best = best_mode_summary(current_summary)
    zero_best = best_mode_summary(official_zero_init_summary)
    current_raw = float(current_summary.get('raw_max', {}).get('image_auroc', 0.0))
    zero_raw = float(official_zero_init_summary.get('raw_max', {}).get('image_auroc', 0.0))
    zero_gain = zero_raw - current_raw
    reasons = [
        f'current best mode={current_best["mode"]} ({current_best["image_auroc"]:.4f})',
        f'current gain vs raw_max={current_best["gain_vs_baseline"]:+.4f}',
        f'official_zero_init raw_max delta={zero_gain:+.4f}',
    ]

    if current_best['family'] == 'smooth' and (
        current_best['gain_vs_baseline'] >= 0.05
        or zero_gain <= -0.05
        or (
            current_best['gain_vs_baseline'] >= 0.01
            and zero_gain <= 0.0
            and current_raw < 0.8
        )
    ):
        suspected = 'score-ranking drift'
        confidence = 'high' if (
            current_best['gain_vs_baseline'] >= 0.1 or zero_gain <= -0.1
        ) else 'medium'
        reasons.append('smooth-family score modes outperform raw_max enough to indicate spike-driven ranking drift')
    elif zero_gain >= 0.05:
        suspected = 'coreset drift'
        confidence = 'high' if zero_gain >= 0.1 else 'medium'
        reasons.append('official_zero_init coreset materially improves raw_max without changing strict score aggregation')
    elif current_raw < 0.5:
        suspected = 'feature drift'
        confidence = 'medium'
        reasons.append('raw_max remains collapsed even after score-side and coreset-side probes')
    else:
        suspected = 'feature drift'
        confidence = 'low'
        reasons.append('diagnose evidence is insufficient to blame score aggregation or coreset drift')

    return {
        'evidence_type': 'diagnose-only',
        'current_best_mode': current_best,
        'official_zero_init_best_mode': zero_best,
        'current_raw_max_auroc': current_raw,
        'official_zero_init_raw_max_auroc': zero_raw,
        'official_zero_init_gain_vs_current_raw_max': zero_gain,
        'suspected_primary_cause': suspected,
        'confidence': confidence,
        'reasons': reasons,
    }


def classify_graphcore_official_compare_root_cause(
    official_image_auroc: float,
    baoiad_image_auroc: float,
    score_delta_stats: Mapping[str, Any],
    bank_compare: Mapping[str, Any] | None = None,
    feature_previews: Mapping[str, Mapping[str, Any]] | None = None,
    train_order_compare: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    gap = float(baoiad_image_auroc) - float(official_image_auroc)
    abs_gap = abs(gap)
    bank_compare = dict(bank_compare or {})
    train_order_compare = dict(train_order_compare or {})
    bank_nn_mean = max(
        float(bank_compare.get('official_to_baoiad_nn', {}).get('mean', 0.0)),
        float(bank_compare.get('baoiad_to_official_nn', {}).get('mean', 0.0)),
    )
    feature_delta_summary = summarize_feature_preview_deltas(feature_previews)
    embedding_std_mean = float(feature_delta_summary['embedding_std_abs_diff']['mean'])
    reasons = [
        f'image_auroc gap={gap:+.4f}',
        f'score_delta_std={float(score_delta_stats.get("std", 0.0)):.4f}',
    ]
    if bank_nn_mean > 0:
        reasons.append(f'bank_nn_mean={bank_nn_mean:.4f}')
    if feature_delta_summary['num_preview_pairs'] > 0:
        reasons.append(f'embedding_std_abs_diff_mean={embedding_std_mean:.4f}')
    if train_order_compare:
        reasons.append(
            'train_order prefix_ratio='
            f'{float(train_order_compare.get("prefix_match_ratio", 0.0)):.4f}'
        )

    if official_image_auroc < 0.65 and abs_gap <= 0.07:
        suspected = 'official instability'
        confidence = 'medium'
        reasons.append('official and BaoIAD are both weak on this class and remain close enough to suspect unstable reference behaviour')
    elif (
        abs_gap >= 0.1 and bank_nn_mean >= 20.0
    ) or (
        feature_delta_summary['num_preview_pairs'] > 0 and embedding_std_mean >= 0.1
    ):
        suspected = 'feature drift'
        confidence = 'high' if abs_gap >= 0.2 else 'medium'
        reasons.append('official compare shows a large representation or memory-bank mismatch')
    elif train_order_compare and not bool(train_order_compare.get('exact_match', False)):
        suspected = 'preprocess/order drift'
        confidence = 'medium' if abs_gap >= 0.04 else 'low'
        reasons.append('official compare found a train-order mismatch before scoring parity was restored')
    elif abs_gap >= 0.04:
        suspected = 'preprocess/order drift'
        confidence = 'low' if abs_gap < 0.08 else 'medium'
        reasons.append('official compare shows a persistent output gap without a cleaner feature/coreset diagnosis')
    else:
        suspected = 'official instability'
        confidence = 'low'
        reasons.append('gap is small enough that official-side instability remains plausible')

    return {
        'evidence_type': 'strict-mainline-vs-official',
        'official_image_auroc': float(official_image_auroc),
        'baoiad_image_auroc': float(baoiad_image_auroc),
        'image_auroc_gap': gap,
        'bank_nn_mean': bank_nn_mean,
        'train_order_compare': train_order_compare,
        'feature_delta_summary': feature_delta_summary,
        'suspected_primary_cause': suspected,
        'confidence': confidence,
        'reasons': reasons,
    }


def merge_graphcore_root_cause_analyses(
    diagnose_analysis: Mapping[str, Any],
    compare_analysis: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    diagnose = dict(diagnose_analysis)
    compare = dict(compare_analysis or {})
    diagnose_cause = diagnose.get('suspected_primary_cause', 'feature drift')
    compare_cause = compare.get('suspected_primary_cause')

    if diagnose_cause in {'score-ranking drift', 'coreset drift'} and diagnose.get('confidence') != 'low':
        suspected = diagnose_cause
        reasons = list(diagnose.get('reasons', []))
    elif compare_cause in GRAPHCORE_ROOT_CAUSES:
        suspected = str(compare_cause)
        reasons = list(compare.get('reasons', []))
    else:
        suspected = str(diagnose_cause)
        reasons = list(diagnose.get('reasons', []))

    if compare:
        reasons.extend(
            reason for reason in compare.get('reasons', [])
            if reason not in reasons
        )

    return {
        'evidence_type': 'merged-targeted-evidence',
        'suspected_primary_cause': suspected,
        'diagnose_cause': diagnose_cause,
        'compare_cause': compare_cause,
        'reasons': reasons,
    }

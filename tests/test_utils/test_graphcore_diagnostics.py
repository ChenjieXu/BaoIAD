"""Tests for GraphCore diagnostic helpers."""

import numpy as np
import pytest
import torch

from baoiad.utils.graphcore_diagnostics import (
    best_mode_summary,
    classify_graphcore_diagnose_root_cause,
    classify_graphcore_official_compare_root_cause,
    image_score_from_maps,
    jaccard_index,
    merge_graphcore_root_cause_analyses,
    safe_ap,
    safe_auroc,
    safe_fpr_at_tpr,
    stats_dict,
    summarize_scores,
    summarize_feature_preview_deltas,
    tensor_stats,
    vector_norm_stats,
)


def test_safe_metrics_handle_binary_scores():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert safe_auroc(labels, scores) == pytest.approx(1.0)
    assert safe_ap(labels, scores) == pytest.approx(1.0)
    assert safe_fpr_at_tpr(labels, scores) == pytest.approx(0.0)


def test_stats_dict_reports_percentiles():
    summary = stats_dict([1.0, 2.0, 3.0, 4.0])
    assert summary['count'] == 4.0
    assert summary['mean'] == pytest.approx(2.5)
    assert summary['p50'] == pytest.approx(2.5)


def test_tensor_and_norm_stats_report_expected_values():
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    summary = tensor_stats(tensor)
    assert summary['shape'] == [2, 2]
    assert summary['mean'] == pytest.approx(2.5)

    norms = vector_norm_stats(np.array([[3.0, 4.0], [5.0, 12.0]], dtype=np.float32))
    assert norms['mean'] == pytest.approx(9.0)


def test_image_score_from_maps_supports_raw_and_smooth_modes():
    patch_map = np.array([[0.0, 1.0], [2.0, 4.0]], dtype=np.float32)
    smooth_map = np.array([[1.0, 2.0], [3.0, 5.0]], dtype=np.float32)

    assert image_score_from_maps(patch_map, smooth_map, 'raw_max') == pytest.approx(4.0)
    assert image_score_from_maps(patch_map, smooth_map, 'raw_mean') == pytest.approx(1.75)
    assert image_score_from_maps(patch_map, smooth_map, 'smooth_max') == pytest.approx(5.0)
    assert image_score_from_maps(patch_map, smooth_map, 'smooth_p95') == pytest.approx(np.percentile(smooth_map, 95))


def test_summarize_scores_reports_class_split_stats():
    labels = [0, 0, 1, 1]
    scores = [0.2, 0.1, 0.8, 0.9]
    summary = summarize_scores(labels, scores)
    assert summary['image_auroc'] == pytest.approx(1.0)
    assert summary['normal']['mean'] == pytest.approx(0.15)
    assert summary['anomaly']['mean'] == pytest.approx(0.85)
    assert summary['gap_mean'] > 0


def test_jaccard_index_handles_overlap():
    assert jaccard_index([1, 2, 3], [2, 3, 4]) == pytest.approx(0.5)


def test_best_mode_summary_reports_smooth_gain_against_raw_max():
    summary = best_mode_summary(
        {
            'raw_max': {'image_auroc': 0.6},
            'smooth_p95': {'image_auroc': 0.82},
            'raw_mean': {'image_auroc': 0.75},
        }
    )

    assert summary['mode'] == 'smooth_p95'
    assert summary['family'] == 'smooth'
    assert summary['gain_vs_baseline'] == pytest.approx(0.22)


def test_summarize_feature_preview_deltas_reports_embedding_and_hook_deltas():
    summary = summarize_feature_preview_deltas(
        {
            'sample.png': {
                'official': {
                    'hook_stats': [{'mean': 1.0, 'std': 2.0}],
                    'embedding_stats': {'mean': 3.0, 'std': 4.0},
                },
                'baoiad': {
                    'hook_stats': [{'mean': 1.5, 'std': 1.0}],
                    'embedding_stats': {'mean': 2.0, 'std': 4.5},
                },
            }
        }
    )

    assert summary['num_preview_pairs'] == 1
    assert summary['hook_mean_abs_diff']['mean'] == pytest.approx(0.5)
    assert summary['embedding_std_abs_diff']['mean'] == pytest.approx(0.5)


def test_classify_graphcore_diagnose_root_cause_prefers_score_ranking():
    analysis = classify_graphcore_diagnose_root_cause(
        current_summary={
            'raw_max': {'image_auroc': 0.6067},
            'smooth_p95': {'image_auroc': 0.82},
            'smooth_max': {'image_auroc': 0.6592},
        },
        official_zero_init_summary={
            'raw_max': {'image_auroc': 0.5196},
            'smooth_p95': {'image_auroc': 0.7117},
        },
    )

    assert analysis['suspected_primary_cause'] == 'score-ranking drift'
    assert analysis['current_best_mode']['mode'] == 'smooth_p95'


def test_classify_graphcore_diagnose_root_cause_prefers_coreset_when_zero_init_helps():
    analysis = classify_graphcore_diagnose_root_cause(
        current_summary={
            'raw_max': {'image_auroc': 0.272},
            'smooth_max': {'image_auroc': 0.3152},
        },
        official_zero_init_summary={
            'raw_max': {'image_auroc': 0.3601},
            'smooth_max': {'image_auroc': 0.4052},
        },
    )

    assert analysis['suspected_primary_cause'] == 'coreset drift'
    assert analysis['official_zero_init_gain_vs_current_raw_max'] == pytest.approx(0.0881)


def test_classify_graphcore_official_compare_root_cause_detects_feature_drift():
    analysis = classify_graphcore_official_compare_root_cause(
        official_image_auroc=0.60,
        baoiad_image_auroc=0.38,
        score_delta_stats={'std': 254.0},
        bank_compare={
            'official_to_baoiad_nn': {'mean': 30.0},
            'baoiad_to_official_nn': {'mean': 29.5},
        },
        feature_previews=None,
    )

    assert analysis['suspected_primary_cause'] == 'feature drift'
    assert analysis['image_auroc_gap'] == pytest.approx(-0.22)


def test_classify_graphcore_official_compare_root_cause_uses_order_mismatch_when_available():
    analysis = classify_graphcore_official_compare_root_cause(
        official_image_auroc=0.70,
        baoiad_image_auroc=0.62,
        score_delta_stats={'std': 312.0},
        train_order_compare={
            'exact_match': False,
            'prefix_match_ratio': 0.0,
        },
    )

    assert analysis['suspected_primary_cause'] == 'preprocess/order drift'
    assert analysis['train_order_compare']['exact_match'] is False


def test_merge_graphcore_root_cause_analyses_prefers_score_side_diagnose_signal():
    merged = merge_graphcore_root_cause_analyses(
        diagnose_analysis={
            'suspected_primary_cause': 'score-ranking drift',
            'confidence': 'medium',
            'reasons': ['smooth modes win'],
        },
        compare_analysis={
            'suspected_primary_cause': 'preprocess/order drift',
            'confidence': 'medium',
            'reasons': ['official gap persists'],
        },
    )

    assert merged['suspected_primary_cause'] == 'score-ranking drift'
    assert 'smooth modes win' in merged['reasons']

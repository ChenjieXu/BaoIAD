"""Tests for SAA diagnostic helpers."""

import pytest
import torch

from baoiad.utils.saa_diagnostics import (
    aggregate_image_score,
    build_image_score_summary,
    image_scores_from_maps,
    minmax_normalize_scores,
    normalize_score_maps,
    records_to_csv_rows,
    resolve_saa_object_controls,
    select_topk_indices,
    summarize_saa_object_path,
)


def test_normalize_score_maps_matches_global_minmax():
    maps = torch.tensor([
        [[[2.0, 4.0], [0.0, 1.0]]],
        [[[3.0, 5.0], [1.0, 1.0]]],
    ])
    normalized = normalize_score_maps(maps)
    expected = torch.tensor([
        [[[0.4, 0.8], [0.0, 0.2]]],
        [[[0.6, 1.0], [0.2, 0.2]]],
    ])
    assert torch.allclose(normalized, expected)


def test_image_scores_from_maps_uses_max():
    maps = torch.tensor([
        [[[0.1, 0.2], [0.3, 0.0]]],
        [[[0.2, 0.1], [0.4, 0.3]]],
    ])
    scores = image_scores_from_maps(maps)
    assert scores.tolist() == pytest.approx([0.3, 0.4])


def test_build_image_score_summary_reports_raw_and_normalized_metrics():
    labels = [0, 0, 1, 1]
    raw_scores = [0.1, 0.2, 0.8, 0.9]
    normalized_scores = [0.0, 0.125, 0.875, 1.0]
    summary = build_image_score_summary(labels, raw_scores, normalized_scores)

    assert summary['num_samples'] == 4
    assert summary['raw']['image_auroc'] == 1.0
    assert summary['normalized']['image_auroc'] == 1.0
    assert summary['raw']['gap_mean'] > 0
    assert summary['normalized']['gap_boundary'] > 0


def test_minmax_normalize_scores_matches_expected_values():
    scores = torch.tensor([2.0, 4.0, 6.0])
    normalized = minmax_normalize_scores(scores)
    assert normalized.tolist() == pytest.approx([0.0, 0.5, 1.0])


def test_aggregate_image_score_supports_multiple_modes():
    amap = torch.tensor([[[0.0, 0.2], [0.4, 0.8]]])
    topk_scores = torch.tensor([0.1, 0.7, 0.5])
    assert aggregate_image_score(amap, mode='map_max') == pytest.approx(0.8)
    assert aggregate_image_score(amap, mode='map_p99') == pytest.approx(torch.quantile(amap.view(-1), 0.99).item())
    assert aggregate_image_score(amap, mode='support_mean') == pytest.approx((0.2 + 0.4 + 0.8) / 3)
    assert aggregate_image_score(amap, mode='topk_combined_score_max', topk_scores=topk_scores) == pytest.approx(0.7)
    assert aggregate_image_score(amap, mode='topk_combined_score_mean', topk_scores=topk_scores) == pytest.approx((0.1 + 0.7 + 0.5) / 3)


def test_select_topk_indices_uses_rank_scores():
    rank_scores = torch.tensor([0.2, 0.8, 0.5, 0.7])
    indices = select_topk_indices(rank_scores, 2)
    assert indices.tolist() == [1, 3]


def test_records_to_csv_rows_flattens_records():
    rows = records_to_csv_rows([
        dict(
            index=0,
            img_path='a.png',
            defect_type='good',
            gt_label=0,
            raw_image_score=0.1,
            normalized_image_score=0.2,
            raw_map_max=0.1,
            raw_map_mean=0.01,
            raw_map_p99=0.1,
            object_area=0.5,
            defect_max_area=0.25,
            object_number=2,
            object_mask_count=3,
            object_mask_non_empty_count=2,
            saliency_strategy='multi',
            num_prompts=4,
            num_boxes=10,
            det_score_max=0.7,
            det_score_mean=0.3,
            saliency_score_max=1.1,
            saliency_score_mean=1.0,
            combined_score_max=0.8,
            combined_score_mean=0.35,
            aggregation_mode='map_max',
            topk_rank_mode='combined',
            image_score_rank_mode='det',
            saliency_score_mode='clipped_multiply',
            saliency_score_clip_max=1.25,
            image_score_area_range=[None, 0.02],
            selected_image_score_count=1,
            selected_image_score_area_ratio_max=0.015,
            selected_image_score_area_ratio_mean=0.015,
            selected_rank_score_max=0.8,
            selected_rank_score_mean=0.4,
            top_phrases=['a', 'b'],
            image_score_phrases=['b'],
        )
    ])

    assert rows == [{
        'index': 0,
        'img_path': 'a.png',
        'defect_type': 'good',
        'gt_label': 0,
        'raw_image_score': 0.1,
        'normalized_image_score': 0.2,
        'raw_map_max': 0.1,
        'raw_map_mean': 0.01,
        'raw_map_p99': 0.1,
        'object_area': 0.5,
        'defect_max_area': 0.25,
        'object_number': 2,
        'object_mask_count': 3,
        'object_mask_non_empty_count': 2,
        'saliency_strategy': 'multi',
        'num_prompts': 4,
        'num_boxes': 10,
        'det_score_max': 0.7,
        'det_score_mean': 0.3,
        'saliency_score_max': 1.1,
        'saliency_score_mean': 1.0,
        'combined_score_max': 0.8,
        'combined_score_mean': 0.35,
        'aggregation_mode': 'map_max',
        'topk_rank_mode': 'combined',
        'image_score_rank_mode': 'det',
        'saliency_score_mode': 'clipped_multiply',
        'saliency_score_clip_max': 1.25,
        'image_score_area_range': [None, 0.02],
        'selected_image_score_count': 1,
        'selected_image_score_area_ratio_max': 0.015,
        'selected_image_score_area_ratio_mean': 0.015,
        'selected_rank_score_max': 0.8,
        'selected_rank_score_mean': 0.4,
        'top_phrases': 'a|b',
        'image_score_phrases': 'b',
    }]


def test_resolve_saa_object_controls_parses_property_prompt():
    controls = resolve_saa_object_controls(
        'pill',
        'the image of pill have 3 dissimilar pill, with a maximum of 7 anomaly. '
        'The anomaly would not exceed 0.4 object area. ',
        default_k_mask=5,
        default_defect_area_threshold=1.0,
    )

    assert controls == {
        'property_prompt': 'the image of pill have 3 dissimilar pill, with a maximum of 7 anomaly. '
        'The anomaly would not exceed 0.4 object area. ',
        'object_prompt': 'pill',
        'object_number': 3,
        'object_max_area': pytest.approx(1.0 / 3.0),
        'k_mask': 7,
        'defect_area_threshold': pytest.approx(0.4),
    }


def test_summarize_saa_object_path_reports_multi_and_fallback():
    multi = summarize_saa_object_path(
        torch.tensor([
            [[1.0, 0.0], [0.0, 0.0]],
            [[0.0, 1.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ]),
        object_number=2,
    )
    fallback = summarize_saa_object_path(
        torch.tensor([
            [[1.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ]),
        object_number=2,
    )

    assert multi == {
        'object_number': 2,
        'object_mask_count': 3,
        'object_mask_non_empty_count': 2,
        'saliency_strategy': 'multi',
    }
    assert fallback == {
        'object_number': 2,
        'object_mask_count': 2,
        'object_mask_non_empty_count': 1,
        'saliency_strategy': 'single_fallback',
    }

"""Tests for vanilla-SAA candidate-sweep helpers."""

import pytest

from baoiad.utils.saa_candidate_sweep import (
    build_candidate_variants,
    build_prompt_pools,
    resolve_prompt_pool_names,
    summarize_candidate_scores,
)


def test_build_prompt_pools_for_pill_exposes_recovery_candidates():
    pools = build_prompt_pools('pill')

    assert pools['general_only'] == [
        ('defect on pill', 'pill'),
        ('damage on pill', 'pill'),
        ('flaw on pill', 'pill'),
    ]
    assert pools['manual_only'] == [
        ('red defect. yellow defect. blue defect. crack. scratch.', 'pill'),
    ]
    assert pools['imprint_only'] == [('imprint', 'pill')]
    assert pools['general_plus_manual'] == [
        ('defect on pill', 'pill'),
        ('damage on pill', 'pill'),
        ('flaw on pill', 'pill'),
        ('red defect. yellow defect. blue defect. crack. scratch.', 'pill'),
    ]
    assert pools['general_plus_manual_plus_imprint'][-1] == ('imprint', 'pill')


def test_resolve_prompt_pool_names_accepts_historical_aliases():
    resolved = resolve_prompt_pool_names('pill', ['general', 'manual_old', 'imprint_only'])

    assert resolved == [
        'general_only',
        'general_plus_manual',
        'imprint_only',
    ]


def test_resolve_prompt_pool_names_skips_missing_defaults_for_non_manual_class():
    resolved = resolve_prompt_pool_names('unknown thing')

    assert resolved == [
        'general_only',
        'imprint_only',
    ]


def test_build_candidate_variants_formats_matrix_entries():
    variants = build_candidate_variants(
        cls_name='pill',
        prompt_pools=['imprint_only'],
        k_masks=[1],
        defect_area_thresholds=[0.5],
        sam_input_modes=['bgr_sam'],
    )

    assert variants == [{
        'name': 'imprint_only_k1_area0p5_bgr_sam',
        'cls_name': 'pill',
        'prompt_pool_name': 'imprint_only',
        'prompts': [('imprint', 'pill')],
        'num_prompts': 1,
        'k_mask': 1,
        'defect_area_threshold': 0.5,
        'sam_input_mode': 'bgr_sam',
        'sam_preconvert_rgb': False,
    }]


def test_build_candidate_variants_rejects_unknown_prompt_pool():
    with pytest.raises(ValueError, match='Prompt pool'):
        build_candidate_variants(
            cls_name='pill',
            prompt_pools=['does_not_exist'],
            k_masks=[1],
            defect_area_thresholds=[0.5],
            sam_input_modes=['bgr_sam'],
        )


def test_summarize_candidate_scores_reports_overall_and_per_type_metrics():
    summary = summarize_candidate_scores({
        'good': [0.1, 0.2],
        'contamination': [0.3, 0.4],
        'faulty_imprint': [0.5, 0.6],
    })

    assert summary['num_samples'] == 6
    assert summary['counts'] == {
        'good': 2,
        'contamination': 2,
        'faulty_imprint': 2,
    }
    assert summary['summary']['image_auroc'] == 1.0
    assert summary['summary']['normal_mean'] == pytest.approx(0.15)
    assert summary['summary']['anomaly_mean'] == pytest.approx(0.45)
    assert summary['per_type']['contamination']['image_auroc'] == 1.0
    assert summary['per_type']['contamination']['anomaly_mean'] == pytest.approx(0.35)
    assert summary['per_type']['faulty_imprint']['image_auroc'] == 1.0
    assert summary['per_type']['faulty_imprint']['anomaly_mean'] == pytest.approx(0.55)

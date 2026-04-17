"""Tests for GraphCore strict-alignment guards."""

import json
from pathlib import Path

from baoiad.utils.graphcore_alignment import (
    GRAPHCORE_STRICT_CORESET_INITIAL_INDEX,
    GRAPHCORE_STRICT_IMAGE_SCORE_MODE,
    graphcore_compare_report_summary,
    graphcore_explicit_order_cfg_overrides,
    graphcore_order_file,
    graphcore_select_compare_reports,
    graphcore_strict_closure_violations,
    graphcore_strict_alignment_violations,
)


def test_graphcore_strict_alignment_accepts_mainline_knobs():
    violations = graphcore_strict_alignment_violations(
        {
            'image_score_mode': GRAPHCORE_STRICT_IMAGE_SCORE_MODE,
            'image_score_mode_overrides': {},
            'coreset_initial_index': GRAPHCORE_STRICT_CORESET_INITIAL_INDEX,
        }
    )

    assert violations == []


def test_graphcore_strict_alignment_rejects_diagnose_only_knobs():
    violations = graphcore_strict_alignment_violations(
        {
            'image_score_mode': 'smooth_p95',
            'image_score_mode_overrides': {'transistor': 'smooth_p95'},
            'coreset_initial_index': 7,
        }
    )

    assert any('image_score_mode' in item for item in violations)
    assert any('image_score_mode_overrides' in item for item in violations)
    assert any('coreset_initial_index' in item for item in violations)


def test_graphcore_order_file_uses_default_alignment_dir():
    path = graphcore_order_file('transistor')
    assert path == Path('runs/alignment/graphcore_orders/transistor.json')


def test_graphcore_explicit_order_cfg_overrides_require_existing_file(tmp_path):
    assert graphcore_explicit_order_cfg_overrides('transistor', order_root=str(tmp_path)) == {}

    order_file = tmp_path / 'transistor.json'
    order_file.write_text('{"indices": [0, 1]}', encoding='utf-8')
    overrides = graphcore_explicit_order_cfg_overrides('transistor', order_root=str(tmp_path))

    assert overrides == {
        'train_dataloader.sampler.type': 'ExplicitOrderSampler',
        'train_dataloader.sampler.index_file': str(order_file),
        'train_dataloader.sampler.round_up': False,
    }


def test_graphcore_select_compare_reports_prefers_exact_order_then_pixel_metrics(tmp_path):
    stale = tmp_path / 'graphcore_official_compare_transistor_stale.json'
    stale.write_text(
        json.dumps(
            {
                'class_name': 'transistor',
                'official': {'image_auroc': 0.70, 'pixel_auroc': 0.90, 'aupro': 0.80},
                'baoiad': {'image_auroc': 0.69, 'pixel_auroc': 0.89, 'aupro': 0.79},
                'train_order_compare': {'exact_match': False},
            }
        ),
        encoding='utf-8',
    )
    exact_no_pixel = tmp_path / 'graphcore_official_compare_transistor_exact.json'
    exact_no_pixel.write_text(
        json.dumps(
            {
                'class_name': 'transistor',
                'official': {'image_auroc': 0.70},
                'baoiad': {'image_auroc': 0.70},
                'train_order_compare': {'exact_match': True},
            }
        ),
        encoding='utf-8',
    )
    exact_with_pixel = tmp_path / 'graphcore_official_compare_transistor_exact_v2.json'
    exact_with_pixel.write_text(
        json.dumps(
            {
                'class_name': 'transistor',
                'official': {'image_auroc': 0.70, 'pixel_auroc': 0.91, 'aupro': 0.81},
                'baoiad': {'image_auroc': 0.70, 'pixel_auroc': 0.91, 'aupro': 0.81},
                'train_order_compare': {'exact_match': True},
            }
        ),
        encoding='utf-8',
    )

    selected = graphcore_select_compare_reports(
        report_root=tmp_path,
        class_names=('transistor',),
    )

    assert selected['transistor']['path'] == str(exact_with_pixel)
    assert selected['transistor']['train_order_exact'] is True
    assert selected['transistor']['has_pixel_metrics'] is True


def test_graphcore_select_compare_reports_prefers_smaller_gap_with_same_protocol(tmp_path):
    stale = tmp_path / 'graphcore_official_compare_grid_old.json'
    stale.write_text(
        json.dumps(
            {
                'class_name': 'grid',
                'official': {'image_auroc': 0.55},
                'baoiad': {'image_auroc': 0.62},
                'train_order_compare': {'exact_match': True},
            }
        ),
        encoding='utf-8',
    )
    better = tmp_path / 'graphcore_official_compare_grid_new.json'
    better.write_text(
        json.dumps(
            {
                'class_name': 'grid',
                'official': {'image_auroc': 0.55},
                'baoiad': {'image_auroc': 0.55},
                'train_order_compare': {'exact_match': True},
            }
        ),
        encoding='utf-8',
    )

    selected = graphcore_select_compare_reports(
        report_root=tmp_path,
        class_names=('grid',),
    )

    assert selected['grid']['path'] == str(better)
    assert selected['grid']['image_gap_abs'] == 0.0


def test_graphcore_compare_report_summary_tracks_missing_pixel_coverage(tmp_path):
    bottle_report = tmp_path / 'graphcore_official_compare_bottle.json'
    bottle_report.write_text(
        json.dumps(
            {
                'class_name': 'bottle',
                'official': {'image_auroc': 0.98, 'pixel_auroc': 0.95, 'aupro': 0.88},
                'baoiad': {'image_auroc': 0.98, 'pixel_auroc': 0.95, 'aupro': 0.88},
                'train_order_compare': {'exact_match': True},
            }
        ),
        encoding='utf-8',
    )
    transistor_report = tmp_path / 'graphcore_official_compare_transistor.json'
    transistor_report.write_text(
        json.dumps(
            {
                'class_name': 'transistor',
                'official': {'image_auroc': 0.70},
                'baoiad': {'image_auroc': 0.70},
                'train_order_compare': {'exact_match': True},
            }
        ),
        encoding='utf-8',
    )

    summary = graphcore_compare_report_summary(
        report_root=tmp_path,
        class_names=('bottle', 'transistor'),
    )

    assert summary['coverage']['reports_found'] == 2
    assert summary['coverage']['missing_reports'] == []
    assert summary['coverage']['missing_exact_order'] == []
    assert summary['coverage']['missing_image_metrics'] == []
    assert summary['coverage']['missing_pixel_metrics'] == ['transistor']
    assert summary['coverage']['strict_closure_ready'] is False


def test_graphcore_strict_closure_violations_require_pixel_metrics(tmp_path):
    report = tmp_path / 'graphcore_official_compare_bottle.json'
    report.write_text(
        json.dumps(
            {
                'class_name': 'bottle',
                'official': {'image_auroc': 0.98},
                'baoiad': {'image_auroc': 0.97},
                'train_order_compare': {'exact_match': True},
            }
        ),
        encoding='utf-8',
    )

    violations = graphcore_strict_closure_violations(
        report_root=tmp_path,
        class_names=('bottle',),
    )

    assert violations == [
        'GraphCore strict closure is missing pixel-side metrics '
        '(pixel_auroc + aupro) in selected compare reports for bottle.'
    ]

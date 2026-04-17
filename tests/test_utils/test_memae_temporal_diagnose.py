"""Tests for MemAE temporal diagnose helpers."""

import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    path = ROOT / 'tools' / 'memae_temporal_diagnose.py'
    spec = importlib.util.spec_from_file_location('baoiad_memae_temporal_diagnose', path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_temporal_std_mean_returns_zero_for_constant_sequence():
    module = _load_module()
    tensor = torch.ones(1, 3, 4, 2, 2)
    value = module._temporal_std_mean(tensor, time_dim=2)
    assert float(value) == pytest.approx(0.0)


def test_group_summary_splits_statistics_by_label():
    module = _load_module()
    records = [
        {'gt_label': 0, 'pred_score': 1.0, 'clip_temporal_std_mean': 0.0, 'encoder_temporal_std_mean': 0.1,
         'attention_temporal_std_mean': 0.2, 'residual_temporal_std_mean': 0.3,
         'spatiotemporal_temporal_std_mean': 0.4, 'anomaly_map_mean': 0.5, 'anomaly_map_max': 0.6},
        {'gt_label': 1, 'pred_score': 2.0, 'clip_temporal_std_mean': 1.0, 'encoder_temporal_std_mean': 1.1,
         'attention_temporal_std_mean': 1.2, 'residual_temporal_std_mean': 1.3,
         'spatiotemporal_temporal_std_mean': 1.4, 'anomaly_map_mean': 1.5, 'anomaly_map_max': 1.6},
    ]

    normal = module._group_summary(records, 0)
    anomaly = module._group_summary(records, 1)

    assert normal['pred_score']['mean'] == pytest.approx(1.0)
    assert anomaly['pred_score']['mean'] == pytest.approx(2.0)
    assert normal['clip_temporal_std_mean']['max'] == pytest.approx(0.0)
    assert anomaly['anomaly_map_max']['mean'] == pytest.approx(1.6)


def test_sample_record_preserves_img_path_and_label():
    module = _load_module()

    class _Sample:
        gt_label = 1
        cls_name = 'bottle'
        defect_type = 'broken'
        img_path = '/tmp/fake.png'

    details = {
        'clip_inputs': torch.zeros(1, 1, 2, 2, 2),
        'encoded': torch.zeros(1, 1, 2, 1, 1),
        'attention': torch.zeros(1, 1, 2, 1, 1),
        'residual': torch.zeros(1, 1, 2, 2, 2),
        'spatiotemporal_map': torch.zeros(1, 2, 2, 2),
        'anomaly_map': torch.zeros(1, 1, 2, 2),
        'img_scores': torch.tensor([0.5]),
    }

    record = module._sample_record(0, details, _Sample())

    assert record['gt_label'] == 1
    assert record['img_path'] == '/tmp/fake.png'
    assert record['pred_score'] == pytest.approx(0.5)

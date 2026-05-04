"""Tests for CutPaste full-model diagnose helpers."""

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    path = ROOT / 'tools' / 'cutpaste_fullmodel_diagnose.py'
    spec = importlib.util.spec_from_file_location('baoiad_cutpaste_fullmodel_diagnose', path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_gap_pool_embeddings_reduces_spatial_dims():
    module = _load_module()
    embeddings = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)

    pooled = module._gap_pool_embeddings(embeddings)

    assert tuple(pooled.shape) == (2, 3)


def test_alignment_summary_reports_identical_vectors():
    module = _load_module()
    lhs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    rhs = lhs.clone()

    summary = module._alignment_summary(lhs, rhs)

    assert summary['same_dim'] is True
    assert summary['cosine_mean'] == 1.0
    assert summary['l2_mean'] == 0.0


def test_resolve_backbone_model_name_accepts_raw_backbone_cfg():
    module = _load_module()

    model_name = module._resolve_backbone_model_name({
        'type': 'RawBackbone',
        'backbone_name': 'resnet18',
    })

    assert model_name == 'resnet18'


def test_build_defect_type_stats_splits_good_and_defects():
    module = _load_module()
    data_list = [
        {'gt_label': 0, 'defect_type': 'good'},
        {'gt_label': 0, 'defect_type': 'good'},
        {'gt_label': 1, 'defect_type': 'scratch'},
        {'gt_label': 1, 'defect_type': 'scratch'},
        {'gt_label': 1, 'defect_type': 'dent'},
    ]
    scores = [0.1, 0.2, 0.8, 0.9, 0.7]

    stats = module._build_defect_type_stats(data_list, scores)

    assert stats['good']['count'] == 2
    assert stats['scratch']['count'] == 2
    assert stats['dent']['count'] == 1
    assert stats['scratch']['score_gap_vs_good'] > 0

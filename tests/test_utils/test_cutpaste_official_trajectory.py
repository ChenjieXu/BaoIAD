"""Tests for CutPaste official trajectory helpers."""

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    path = ROOT / 'tools' / 'cutpaste_official_trajectory.py'
    spec = importlib.util.spec_from_file_location('baoiad_cutpaste_official_trajectory', path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_steps_sorts_and_deduplicates():
    module = _load_module()

    assert module._parse_steps([30, 10, 20, 20]) == [10, 20, 30]


def test_official_to_baoiad_state_dict_maps_expected_prefixes():
    module = _load_module()
    state_dict = {
        'resnet18.conv1.weight': torch.tensor([1.0]),
        'head.0.weight': torch.tensor([2.0]),
        'out.bias': torch.tensor([3.0]),
    }

    converted = module._official_to_baoiad_state_dict(state_dict)

    assert torch.equal(converted['backbone.conv1.weight'], torch.tensor([1.0]))
    assert torch.equal(converted['head.0.weight'], torch.tensor([2.0]))
    assert torch.equal(converted['classifier.bias'], torch.tensor([3.0]))


def test_build_compare_summary_reports_first_stop_line():
    module = _load_module()
    snapshots = [
        {
            'label': 'iter_10',
            'official_checkpoint_path': '/tmp/official_iter_10.pth',
            'baoiad_checkpoint_path': '/tmp/baoiad_iter_10.pth',
            'metrics': {'image_auroc': 0.82, 'image_ap': 0.80, 'image_f1max': 0.75},
            'score_gap': {'score_gap_mean': 0.90, 'normal': {'mean': 0.10}, 'anomaly': {'mean': 1.00}},
        },
        {
            'label': 'iter_20',
            'official_checkpoint_path': '/tmp/official_iter_20.pth',
            'baoiad_checkpoint_path': '/tmp/baoiad_iter_20.pth',
            'metrics': {'image_auroc': 0.17, 'image_ap': 0.20, 'image_f1max': 0.30},
            'score_gap': {'score_gap_mean': -0.20, 'normal': {'mean': 0.80}, 'anomaly': {'mean': 0.60}},
        },
    ]

    compare = module._build_compare_summary(
        class_name='screw',
        reference_root=ROOT / '.refs' / 'pytorch-cutpaste',
        snapshots=snapshots,
    )

    assert compare['source_label'] == 'official_reference'
    assert compare['ordered_checkpoint_labels'] == ['iter_10', 'iter_20']
    assert compare['first_stop_line_label'] == 'iter_20'
    assert compare['checkpoints']['iter_10']['checkpoint_path'] == '/tmp/baoiad_iter_10.pth'

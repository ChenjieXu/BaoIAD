"""Tests for CFlowDetector."""

from pathlib import Path

from mmengine.config import Config
import pytest
import torch
from unittest import TestCase
from baoiad.structures import ADDataSample

import baoiad  # noqa: F401
from baoiad.registry import MODELS


def _make_data_samples(batch_size, H=256, W=256):
    samples = []
    for i in range(batch_size):
        s = ADDataSample()
        s.gt_label = i % 2
        s.gt_mask = torch.zeros(H, W)
        s.cls_name = 'bottle'
        s.img_path = f'/fake/{i}.png'
        s.defect_type = 'good'
        samples.append(s)
    return samples


class TestCFlowDetector(TestCase):
    def setUp(self):
        self.cfg = dict(
            type='CFlowDetector',
            backbone='resnet18',
            condition_dim=64,
            coupling_blocks=2,
            clamp_alpha=1.9,
            fiber_batch_size=64,
        )

    def test_forward_tensor(self):
        model = MODELS.build(self.cfg)
        model.eval()
        out = model(torch.randn(2, 3, 64, 64), mode='tensor')
        assert isinstance(out, (tuple, list, torch.Tensor))

    def test_forward_loss(self):
        model = MODELS.build(self.cfg)
        model.train()
        data_samples = _make_data_samples(2, 64, 64)
        out = model(torch.randn(2, 3, 64, 64), data_samples, mode='loss')
        assert isinstance(out, dict)

    def test_forward_predict(self):
        model = MODELS.build(self.cfg)
        model.eval()
        data_samples = _make_data_samples(2, 64, 64)
        out = model(torch.randn(2, 3, 64, 64), data_samples, mode='predict')
        assert isinstance(out, list)
        assert len(out) == 2

    def test_missing_official_reference_can_be_required(self):
        cfg = dict(self.cfg)
        cfg['reference_repo'] = '/tmp/missing-cflow-ref'
        cfg['require_official_reference'] = True
        with pytest.raises(FileNotFoundError):
            MODELS.build(cfg)


def test_strict_config_matches_official_defaults():
    root = Path(__file__).resolve().parents[3]
    cfg = Config.fromfile(root / 'configs' / 'cflow' / 'cflow_mvtec_strict.py')

    assert cfg.model['backbone']['type'] == 'FeatureExtractor'
    assert cfg.model['backbone']['backbone_name'] == 'wide_resnet50_2'
    assert tuple(cfg.model['backbone']['out_indices']) == (2, 3, 4)
    assert cfg.model['permute_soft'] is True
    assert cfg.model['fiber_batch_size'] == 256
    assert cfg.model['require_official_reference'] is True
    assert cfg.train_cfg['type'] == 'CFlowOfficialTrainLoop'
    assert cfg.train_cfg['max_epochs'] == 25
    assert cfg.train_cfg['sub_epochs'] == 8
    assert cfg.train_dataloader['batch_size'] == 32
    assert cfg.benchmark_result_selector['mode'] == 'best_per_metric'

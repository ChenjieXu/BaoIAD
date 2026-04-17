"""Tests for MambaADDetector."""

import torch
from unittest import TestCase

from baoiad.structures import ADDataSample

import baoiad  # noqa: F401
from baoiad.registry import MODELS


def _make_data_samples(batch_size, H=64, W=64):
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


class TestMambaADDetector(TestCase):
    def setUp(self):
        self.cfg = dict(
            type='MambaADDetector',
            backbone=dict(
                type='TIMMBackbone',
                model_name='resnet18',
                pretrained=False,
                features_only=True,
                out_indices=(1, 2, 3),
                frozen=True,
            ),
            num_blocks=1,
            d_state=4,
            d_conv=4,
            expand=2,
            smooth_sigma=0.0,
        )

    def test_forward_tensor(self):
        model = MODELS.build(self.cfg)
        model.eval()
        out = model(torch.randn(2, 3, 64, 64), mode='tensor')
        assert isinstance(out, list)
        assert len(out) == 3

    def test_forward_loss(self):
        model = MODELS.build(self.cfg)
        model.train()
        data_samples = _make_data_samples(2, 64, 64)
        out = model(torch.randn(2, 3, 64, 64), data_samples, mode='loss')
        assert isinstance(out, dict)
        assert 'loss' in out
        assert torch.isfinite(out['loss'])

    def test_forward_predict(self):
        model = MODELS.build(self.cfg)
        model.eval()
        data_samples = _make_data_samples(2, 64, 64)
        out = model(torch.randn(2, 3, 64, 64), data_samples, mode='predict')
        assert isinstance(out, list)
        assert len(out) == 2
        for s in out:
            assert hasattr(s, 'pred_score')
            assert hasattr(s, 'pred_anomaly_map')

    def test_encoder_frozen(self):
        model = MODELS.build(self.cfg)
        model.train()
        assert not model.encoder.net.training

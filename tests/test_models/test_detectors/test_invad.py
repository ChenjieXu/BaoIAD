"""Tests for InvADDetector."""

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


class TestInvADDetector(TestCase):
    def setUp(self):
        self.cfg = dict(
            type='InvADDetector',
            backbone='resnet18',
            out_cha=64,
            latent_channel_size=32,
        )
        self.H = self.W = 256

    def test_forward_tensor(self):
        model = MODELS.build(self.cfg)
        model.eval()
        out = model(torch.randn(2, 3, self.H, self.W), mode='tensor')
        assert out is not None

    def test_forward_loss(self):
        model = MODELS.build(self.cfg)
        model.train()
        data_samples = _make_data_samples(2, self.H, self.W)
        out = model(torch.randn(2, 3, self.H, self.W), data_samples, mode='loss')
        assert isinstance(out, dict)

    def test_forward_predict(self):
        model = MODELS.build(self.cfg)
        model.eval()
        data_samples = _make_data_samples(2, self.H, self.W)
        out = model(torch.randn(2, 3, self.H, self.W), data_samples, mode='predict')
        assert isinstance(out, list)
        assert len(out) == 2
        for sample in out:
            pred_map = sample.pred_anomaly_map.squeeze(0)
            assert sample.pred_score == pytest.approx(float(pred_map.max()))

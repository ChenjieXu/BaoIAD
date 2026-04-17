"""Tests for PaDiMDetector."""

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

class TestPaDiMDetector(TestCase):
    def setUp(self):
        self.cfg = dict(type='PaDiMDetector', backbone='resnet18', d_reduced=50, eps=0.01)

    def test_forward_tensor(self):
        model = MODELS.build(self.cfg)
        model.eval()
        out = model(torch.randn(2, 3, 64, 64), mode='tensor')
        assert out is not None

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
        # Memory bank methods need collect phase before predict
        # Run a few forward passes in loss mode to populate memory
        for _ in range(3):
            model(torch.randn(2, 3, 64, 64), _make_data_samples(2, 64, 64), mode='loss')
        if hasattr(model, 'fit'):
            model.fit()
        out = model(torch.randn(2, 3, 64, 64), data_samples, mode='predict')
        assert isinstance(out, list)
        assert len(out) == 2

    def test_predict_raises_before_statistics_are_built(self):
        model = MODELS.build(self.cfg)
        model.eval()
        data_samples = _make_data_samples(1, 64, 64)
        with pytest.raises(RuntimeError, match='Gaussian statistics are not built'):
            model(torch.randn(1, 3, 64, 64), data_samples, mode='predict')

    def test_build_memory_bank_from_dataloader(self):
        model = MODELS.build(self.cfg)
        loader = [
            dict(
                inputs=torch.randn(2, 3, 64, 64),
                data_samples=_make_data_samples(2, 64, 64),
            )
        ]
        model.build_memory_bank(loader)

        assert model.mean is not None
        assert model.cov_inv is not None

        outputs = model(torch.randn(1, 3, 64, 64), _make_data_samples(1, 64, 64), mode='predict')
        assert outputs[0].pred_anomaly_map.shape == (1, 64, 64)
        assert torch.isfinite(outputs[0].pred_anomaly_map).all()

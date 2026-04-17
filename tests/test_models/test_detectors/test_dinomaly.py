"""Tests for DinomalyDetector."""

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


import os
from pathlib import Path


def _dinomaly_weights_available():
    """Check if Dinomaly DINOv2 weights are cached or can be downloaded."""
    # Check for cached weights first (avoids slow download)
    cache_dir = Path('./pre_trained')
    weight_file = cache_dir / 'dinov2_vitb14_reg4_pretrain.pth'
    if weight_file.exists():
        return True
    # Try a quick network check
    try:
        import urllib.request
        urllib.request.urlopen(
            'https://dl.fbaipublicfiles.com/dinov2/', timeout=5
        )
        return True
    except Exception:
        return False


_SKIP_DINOMALY = not _dinomaly_weights_available()

@pytest.mark.skipif(_SKIP_DINOMALY, reason='DINOv2 weights unavailable (no cache or network)')
class TestDinomalyDetector(TestCase):
    def setUp(self):
        self.cfg = dict(type='DinomalyDetector', encoder_name='dinov2reg_vit_base_14', bottleneck_dropout=0.2, decoder_depth=8)

    def test_forward_tensor(self):
        model = MODELS.build(self.cfg)
        model.eval()
        out = model(torch.randn(2, 3, 224, 224), mode='tensor')
        assert out is not None

    def test_forward_loss(self):
        model = MODELS.build(self.cfg)
        model.train()
        data_samples = _make_data_samples(2, 224, 224)
        out = model(torch.randn(2, 3, 224, 224), data_samples, mode='loss')
        assert isinstance(out, dict)

    def test_forward_predict(self):
        model = MODELS.build(self.cfg)
        model.eval()
        data_samples = _make_data_samples(2, 224, 224)
        out = model(torch.randn(2, 3, 224, 224), data_samples, mode='predict')
        assert isinstance(out, list)
        assert len(out) == 2

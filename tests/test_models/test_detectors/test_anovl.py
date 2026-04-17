"""Tests for AnoVLDetector."""

import os
import pytest
import torch
import torch.nn as nn
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


try:
    import open_clip  # noqa: F401
    HAS_OPEN_CLIP = True
except ImportError:
    HAS_OPEN_CLIP = False


LOCAL_PRETRAINED = 'pretrained/open_clip/vit_b_16_plus_240-laion400m_e32-699c4b84.pt'
HAS_LOCAL_PRETRAINED = os.path.exists(LOCAL_PRETRAINED)


@pytest.mark.skipif(
    not HAS_OPEN_CLIP or not HAS_LOCAL_PRETRAINED,
    reason='open_clip or local AnoVL weights unavailable',
)
class TestAnoVLDetector(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = dict(
            type='AnoVLDetector',
            clip_model='ViT-B-16-plus-240',
            pretrained=LOCAL_PRETRAINED,
            image_size=64,
            tta_enabled=False,
        )
        cls.model = MODELS.build(cls.cfg)

    def test_forward_tensor(self):
        self.model.eval()
        out = self.model(torch.randn(2, 3, 64, 64), mode='tensor')
        assert out is not None
        assert out.ndim == 3  # (B, N, D)
        assert out.shape[0] == 2

    def test_forward_loss(self):
        self.model.train()
        data_samples = _make_data_samples(2, 64, 64)
        out = self.model(torch.randn(2, 3, 64, 64), data_samples, mode='loss')
        assert isinstance(out, dict)
        assert 'loss' in out

    def test_forward_predict(self):
        self.model.eval()
        data_samples = _make_data_samples(2, 64, 64)
        out = self.model(torch.randn(2, 3, 64, 64), data_samples, mode='predict')
        assert isinstance(out, list)
        assert len(out) == 2
        assert hasattr(out[0], 'pred_score')
        assert hasattr(out[0], 'pred_anomaly_map')
        assert out[0].pred_anomaly_map.shape[-2:] == (64, 64)


@pytest.mark.skipif(
    not HAS_OPEN_CLIP or not HAS_LOCAL_PRETRAINED,
    reason='open_clip or local AnoVL weights unavailable',
)
class TestAnoVLDetectorTTA(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = dict(
            type='AnoVLDetector',
            clip_model='ViT-B-16-plus-240',
            pretrained=LOCAL_PRETRAINED,
            image_size=64,
            tta_enabled=True,
            tta_epochs=2,
        )
        cls.model = MODELS.build(cls.cfg)

    def test_forward_predict_tta(self):
        self.model.eval()
        data_samples = _make_data_samples(1, 64, 64)
        out = self.model(torch.randn(1, 3, 64, 64), data_samples, mode='predict')
        assert isinstance(out, list)
        assert len(out) == 1
        assert hasattr(out[0], 'pred_score')
        assert hasattr(out[0], 'pred_anomaly_map')
        assert out[0].pred_anomaly_map.shape[-2:] == (64, 64)


@pytest.mark.skipif(
    not HAS_OPEN_CLIP or not HAS_LOCAL_PRETRAINED,
    reason='open_clip or local AnoVL weights unavailable',
)
def test_forward_predict_uses_per_sample_class_prompts(monkeypatch):
    model = MODELS.build(
        dict(
            type='AnoVLDetector',
            clip_model='ViT-B-16-plus-240',
            pretrained=LOCAL_PRETRAINED,
            image_size=64,
            tta_enabled=False,
        ))
    model.eval()
    model.linear_layer = nn.Identity()

    shared_dim = model.shared_dim

    def fake_encode_image_vv(x, out_layers=None):
        cls = torch.zeros(x.shape[0], shared_dim, device=x.device, dtype=x.dtype)
        cls[:, 0] = 1
        patch_layers = []
        if out_layers is None or len(out_layers) > 0:
            patch_layers = [
                torch.zeros(x.shape[0], 4, shared_dim, device=x.device, dtype=x.dtype)
            ]
        return cls, patch_layers

    def fake_get_text_features(cls_name, device, dtype):
        normal = torch.zeros(shared_dim, device=device, dtype=dtype)
        abnormal = torch.zeros(shared_dim, device=device, dtype=dtype)
        if cls_name == 'bottle':
            normal[0] = 1
            abnormal[1] = 1
        else:
            normal[1] = 1
            abnormal[0] = 1
        prompt = torch.stack([normal, abnormal], dim=0)
        return normal, abnormal, prompt

    def fake_compute_anomaly_map(*args, **kwargs):
        return torch.zeros(1, 1, 2, 2)

    monkeypatch.setattr(model, '_encode_image_vv', fake_encode_image_vv)
    monkeypatch.setattr(model, '_get_text_features', fake_get_text_features)
    monkeypatch.setattr(model, '_compute_anomaly_map', fake_compute_anomaly_map)

    data_samples = _make_data_samples(2, 64, 64)
    data_samples[0].cls_name = 'bottle'
    data_samples[1].cls_name = 'capsule'

    out = model(torch.randn(2, 3, 64, 64), data_samples, mode='predict')

    assert out[0].pred_score < 0.5
    assert out[1].pred_score > 0.5

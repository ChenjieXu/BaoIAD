"""Tests for DFKDEDetector."""

import pytest
import torch
from torch import nn
from unittest import TestCase

import baoiad  # noqa: F401
import baoiad.models.backbone_utils as backbone_utils
from baoiad.structures import ADDataSample
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


class _DummySingleLayerBackbone(nn.Module):
    def forward(self, x):
        base = x.mean(dim=1, keepdim=True)
        return [base + 5.0]


class _DummyMultiLayerBackbone(nn.Module):
    def forward(self, x):
        base = x.mean(dim=1, keepdim=True)
        return [base + 1.0, base + 2.0]


class TestDFKDEDetector(TestCase):
    def setUp(self):
        self.cfg = dict(type='DFKDEDetector', backbone='resnet18', n_pca_components=4, max_training_points=100)

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


def test_extract_features_concatenates_all_requested_layers(monkeypatch):
    monkeypatch.setattr(backbone_utils, 'build_feature_extractor', lambda *args, **kwargs: _DummyMultiLayerBackbone())

    model = MODELS.build(dict(type='DFKDEDetector', backbone='resnet18', n_pca_components=2))
    features = model.extract_features(torch.zeros(2, 3, 8, 8))

    assert list(features.shape) == [2, 2]
    torch.testing.assert_close(features[:, 0], torch.full((2,), 1.0))
    torch.testing.assert_close(features[:, 1], torch.full((2,), 2.0))


def test_build_memory_bank_raises_on_empty_bank(monkeypatch):
    monkeypatch.setattr(backbone_utils, 'build_feature_extractor', lambda *args, **kwargs: _DummySingleLayerBackbone())

    model = MODELS.build(dict(type='DFKDEDetector', backbone='resnet18', n_pca_components=2))

    with pytest.raises(ValueError, match='Memory bank is empty'):
        model.build_memory_bank()


def test_build_memory_bank_does_not_shrink_pca_for_small_sample_count(monkeypatch):
    monkeypatch.setattr(backbone_utils, 'build_feature_extractor', lambda *args, **kwargs: _DummySingleLayerBackbone())

    model = MODELS.build(dict(type='DFKDEDetector', backbone='resnet18', n_pca_components=8, max_training_points=100))
    model.train()
    for _ in range(3):
        model(torch.randn(2, 3, 8, 8), _make_data_samples(2, 8, 8), mode='loss')

    model.build_memory_bank()

    assert model.classifier.pca_model.n_components == 8
    assert model.classifier.pca_model.mean is None
    assert len(model._memory_bank) == 0

    model.eval()
    with pytest.raises(Exception):
        model(torch.randn(2, 3, 8, 8), _make_data_samples(2, 8, 8), mode='predict')

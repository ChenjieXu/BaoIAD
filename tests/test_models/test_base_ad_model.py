"""Tests for BaseADModel."""

import pytest
import torch
import torch.nn as nn

import baoiad  # noqa: F401

from baoiad.registry import MODELS
from baoiad.models.base_ad_model import BaseADModel


# Register minimal dummy modules for testing
@MODELS.register_module()
class _DummyBackbone(nn.Module):
    def __init__(self, out_channels=64):
        super().__init__()
        self.conv = nn.Conv2d(3, out_channels, 1)

    def forward(self, x):
        return (self.conv(x),)


@MODELS.register_module()
class _DummyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1)  # dummy param

    def loss(self, feats, data_samples=None):
        return {'loss': feats[0].sum() * 0}

    def predict(self, feats, data_samples=None):
        from baoiad.structures import ADDataSample
        B = feats[0].shape[0]
        results = []
        for i in range(B):
            r = ADDataSample()
            r.pred_score = 0.5
            r.pred_anomaly_map = torch.zeros(1, 8, 8)
            results.append(r)
        return results

    def forward(self, feats):
        return feats


_BACKBONE_CFG = dict(type='_DummyBackbone', out_channels=64)
_HEAD_CFG = dict(type='_DummyHead')


class TestBaseADModel:
    def _build_model(self, freeze=True):
        return BaseADModel(
            backbone=_BACKBONE_CFG,
            head=_HEAD_CFG,
            freeze_backbone=freeze,
        )

    def test_forward_tensor_mode(self):
        model = self._build_model()
        x = torch.randn(2, 3, 32, 32)
        out = model(x, mode='tensor')
        assert isinstance(out, tuple)
        assert out[0].shape[0] == 2

    def test_forward_loss_mode(self):
        model = self._build_model()
        x = torch.randn(2, 3, 32, 32)
        out = model(x, mode='loss')
        assert isinstance(out, dict)
        assert 'loss' in out

    def test_forward_predict_mode(self):
        model = self._build_model()
        x = torch.randn(2, 3, 32, 32)
        out = model(x, mode='predict')
        assert isinstance(out, list)
        assert len(out) == 2

    def test_freeze_backbone(self):
        model = self._build_model(freeze=True)
        for p in model.backbone.parameters():
            assert not p.requires_grad

    def test_unfreeze_backbone(self):
        model = self._build_model(freeze=False)
        for p in model.backbone.parameters():
            assert p.requires_grad

    def test_invalid_mode(self):
        model = self._build_model()
        x = torch.randn(1, 3, 32, 32)
        with pytest.raises(RuntimeError, match='Invalid mode'):
            model(x, mode='invalid')

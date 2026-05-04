"""Tests for PatchCore detector."""

from pathlib import Path

import pytest
import torch
from mmengine import Config

import baoiad  # noqa: F401

from baoiad.models.detectors.patchcore import PatchCore

# Reuse dummy modules registered in test_base_ad_model
from tests.test_models.test_base_ad_model import _DummyBackbone, _DummyHead  # noqa: F401

_CFG = dict(
    backbone=dict(type='_DummyBackbone', out_channels=64),
    neck=None,
    head=dict(type='_DummyHead'),
)

ROOT = Path(__file__).resolve().parents[3]


class TestPatchCore:
    def _build(self):
        return PatchCore(**_CFG)

    def test_forward_tensor(self):
        model = self._build()
        out = model(torch.randn(2, 3, 32, 32), mode='tensor')
        assert isinstance(out, tuple)

    def test_forward_loss(self):
        model = self._build()
        out = model(torch.randn(2, 3, 32, 32), mode='loss')
        assert isinstance(out, dict)

    def test_forward_predict(self):
        model = self._build()
        out = model(torch.randn(2, 3, 32, 32), mode='predict')
        assert isinstance(out, list)

    def test_backbone_frozen(self):
        model = self._build()
        for p in model.backbone.parameters():
            assert not p.requires_grad

    def test_invalid_mode(self):
        model = self._build()
        with pytest.raises(RuntimeError):
            model(torch.randn(1, 3, 32, 32), mode='bad')


def test_patchcore_alignment_config_freezes_reference_settings():
    cfg = Config.fromfile(ROOT / 'configs' / 'patchcore' / 'patchcore_wrn50_256_mvtec_strict.py')
    assert tuple(cfg.model.backbone.out_indices) == (2, 3)
    assert cfg.model.head.num_neighbors == 9
    assert tuple(cfg.model.head.input_size) == (256, 256)
    assert cfg.model.head.blur_sigma == 4.0
    assert cfg.model.head.reweight_scores is False
    assert cfg.model.head.image_score_source == 'postprocessed'
    assert cfg.model.head.patch_score_neighbors == 1
    assert cfg.model.head.patch_score_reduction == 'first'
    assert cfg.model.head.coreset_sampling_method == 'approx_greedy'
    assert cfg.model.head.coreset_projection_dim == 128
    assert cfg.model.head.coreset_starting_points == 10

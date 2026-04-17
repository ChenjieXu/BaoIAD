"""Tests for FREDetector."""

from pathlib import Path
from unittest import TestCase

import pytest
import torch
from mmengine import Config

import baoiad  # noqa: F401
from baoiad.registry import MODELS
from baoiad.structures import ADDataSample

ROOT = Path(__file__).resolve().parents[3]


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


class TestFREDetector(TestCase):
    def setUp(self):
        self.cfg = dict(
            type='FREDetector',
            backbone=dict(
                type='TIMMBackbone',
                model_name='resnet18',
                pretrained=False,
                features_only=True,
                frozen=True,
            ),
            layer='layer3',
            pooling_kernel_size=2,
            input_dim=1024,
            latent_dim=32,
        )

    def test_forward_tensor(self):
        model = MODELS.build(self.cfg)
        model.eval()
        features_in, features_out = model(torch.randn(2, 3, 64, 64), mode='tensor')
        assert tuple(features_in.shape) == (2, 1024)
        assert tuple(features_out.shape) == (2, 1024)
        assert torch.isfinite(features_in).all()
        assert torch.isfinite(features_out).all()

    def test_forward_loss(self):
        model = MODELS.build(self.cfg)
        model.train()
        data_samples = _make_data_samples(2, 64, 64)
        out = model(torch.randn(2, 3, 64, 64), data_samples, mode='loss')
        assert sorted(out.keys()) == ['loss']
        assert torch.isfinite(out['loss']).all()

    def test_forward_predict(self):
        model = MODELS.build(self.cfg)
        model.eval()
        data_samples = _make_data_samples(2, 64, 64)
        out = model(torch.randn(2, 3, 64, 64), data_samples, mode='predict')
        assert isinstance(out, list)
        assert len(out) == 2
        assert all(hasattr(result, 'pred_score') for result in out)
        maps = torch.stack([result.pred_anomaly_map for result in out])
        assert tuple(maps.shape) == (2, 1, 64, 64)
        assert torch.isfinite(maps).all()

    def test_backbone_stays_frozen_in_train_mode(self):
        model = MODELS.build(self.cfg)
        model.train()
        assert model.backbone.training is False
        assert all(not parameter.requires_grad for parameter in model.backbone.parameters())


def test_fre_alignment_config_matches_reference_settings():
    cfg = Config.fromfile(ROOT / 'configs' / 'fre' / 'fre_256_mvtec_strict.py')
    assert cfg.model.backbone.type == 'TIMMBackbone'
    assert cfg.model.backbone.model_name == 'wide_resnet50_2'
    assert cfg.model.backbone.frozen is True
    assert cfg.model.backbone.allow_legacy_fallback is False
    assert cfg.model.layer == 'layer3'
    assert cfg.model.pooling_kernel_size == 4
    assert cfg.model.input_dim == 16384
    assert cfg.model.latent_dim == 220
    assert cfg.optim_wrapper.optimizer.type == 'Adam'
    assert cfg.optim_wrapper.optimizer.lr == 1e-3
    assert cfg.optim_wrapper.optimizer.weight_decay == 0
    assert cfg.param_scheduler == []
    assert cfg.train_cfg.max_epochs == 220
    assert cfg.train_cfg.val_interval == 10
    assert cfg.benchmark_result_selector.mode == 'last'


def test_fre_reference_feature_dim_matches_wrn50_layer3_shape():
    cfg = Config.fromfile(ROOT / 'configs' / 'fre' / 'fre_256_mvtec_strict.py')
    cfg.model.backbone.pretrained = False
    model = MODELS.build(cfg.model)
    model.eval()

    features_in, features_out = model(torch.randn(1, 3, 256, 256), mode='tensor')

    assert tuple(features_in.shape) == (1, cfg.model.input_dim)
    assert tuple(features_out.shape) == (1, cfg.model.input_dim)


def test_fre_multilayer_predict_supports_weighted_fusion_and_topk_score():
    cfg = dict(
        type='FREDetector',
        backbone=dict(
            type='TIMMBackbone',
            model_name='resnet18',
            pretrained=False,
            features_only=True,
            frozen=True,
        ),
        layers=['layer2', 'layer3'],
        pooling_kernel_sizes=[2, 2],
        input_dims=[2048, 1024],
        latent_dims=[16, 8],
        layer_weights=[0.6, 0.4],
        layer_fusion_mode='weighted_sum',
        layer_norm_mode='zscore',
        image_score_mode='topk_mean',
        topk_ratio=0.1,
        loss=dict(type='MSELoss'),
    )
    model = MODELS.build(cfg)
    model.eval()

    features_in, features_out = model(torch.randn(2, 3, 64, 64), mode='tensor')
    assert isinstance(features_in, list)
    assert isinstance(features_out, list)
    assert [tuple(feature.shape) for feature in features_in] == [(2, 2048), (2, 1024)]
    assert [tuple(feature.shape) for feature in features_out] == [(2, 2048), (2, 1024)]

    data_samples = _make_data_samples(2, 64, 64)
    outputs = model(torch.randn(2, 3, 64, 64), data_samples=data_samples, mode='predict')
    assert len(outputs) == 2
    maps = torch.stack([result.pred_anomaly_map for result in outputs])
    scores = torch.tensor([float(result.pred_score) for result in outputs])
    assert tuple(maps.shape) == (2, 1, 64, 64)
    assert torch.isfinite(maps).all()
    assert torch.isfinite(scores).all()


def test_fre_multilayer_config_validates_lengths():
    cfg = dict(
        type='FREDetector',
        backbone=dict(
            type='TIMMBackbone',
            model_name='resnet18',
            pretrained=False,
            features_only=True,
            frozen=True,
        ),
        layers=['layer2', 'layer3'],
        pooling_kernel_sizes=[2],
        input_dims=[2048, 1024],
        latent_dims=[16, 8],
    )
    with pytest.raises(ValueError):
        MODELS.build(cfg)


def test_fre_strict_predict_scores_use_pre_upsample_map_sum():
    cfg = dict(
        type='FREDetector',
        backbone=dict(
            type='TIMMBackbone',
            model_name='resnet18',
            pretrained=False,
            features_only=True,
            frozen=True,
        ),
        layer='layer3',
        pooling_kernel_size=2,
        input_dim=4,
        latent_dim=2,
        image_score_mode='sum',
        layer_norm_mode='none',
    )
    model = MODELS.build(cfg)
    model.eval()

    features_in = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    features_out = torch.zeros_like(features_in)
    feature_shapes = [torch.Size([1, 1, 2, 2])]

    def _fake_get_features(_inputs):
        return [features_in], [features_out], feature_shapes

    model._get_features = _fake_get_features  # type: ignore[method-assign]
    outputs = model(torch.randn(1, 3, 64, 64), data_samples=_make_data_samples(1, 64, 64), mode='predict')

    # Official FRE computes image score from the native pooled residual map:
    # [[1, 4], [9, 16]] -> sum = 30, before interpolation to 64x64.
    assert float(outputs[0].pred_score) == pytest.approx(30.0)
    assert tuple(outputs[0].pred_anomaly_map.shape) == (1, 64, 64)



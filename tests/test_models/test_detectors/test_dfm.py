"""Tests for DFMDetector."""

import torch
from torch import nn

import baoiad  # noqa: F401
from baoiad.models.detectors import dfm as dfm_module
from baoiad.models.detectors.dfm import DFMDetector
from baoiad.structures import ADDataSample


def _make_data_samples(batch_size, H=256, W=256):
    samples = []
    for i in range(batch_size):
        sample = ADDataSample()
        sample.gt_label = i % 2
        sample.gt_mask = torch.zeros(H, W)
        sample.cls_name = 'bottle'
        sample.img_path = f'/fake/{i}.png'
        sample.defect_type = 'good'
        samples.append(sample)
    return samples


class _DummyFeatureExtractor(nn.Module):
    def forward(self, x):
        base = x.mean(dim=1, keepdim=True)
        return [base + 1.0, base + 2.0, base + 3.0, base + 4.0]


class _DummyTimmBackbone(nn.Module):
    def forward(self, x):
        base = x.mean(dim=1, keepdim=True)
        return [base + 7.0]


def test_extract_features_uses_requested_layer(monkeypatch):
    import baoiad.models.backbone_utils as backbone_utils

    monkeypatch.setattr(backbone_utils, 'build_feature_extractor', lambda *args, **kwargs: _DummyFeatureExtractor())

    model = DFMDetector(backbone='resnet18', layer='layer3', pooling_kernel_size=2, pca_level=0.97)
    features, feature_shape = model._extract_features(torch.zeros(2, 3, 8, 8))

    assert list(feature_shape) == [2, 1, 4, 4]
    torch.testing.assert_close(features, torch.full((2, 16), 3.0))


def test_timm_backbone_path_freezes_requested_out_index(monkeypatch):
    captured_cfg = {}

    def _fake_build(cfg):
        captured_cfg.update(cfg)
        return _DummyTimmBackbone()

    monkeypatch.setattr(dfm_module.MODELS, 'build', _fake_build)

    model = DFMDetector(
        backbone=dict(type='TIMMBackbone', model_name='wide_resnet50_2', pretrained=True),
        layer='layer3',
        pooling_kernel_size=1,
        pca_level=0.97,
    )
    features, feature_shape = model._extract_features(torch.zeros(1, 3, 4, 4))

    assert captured_cfg['out_indices'] == (3,)
    assert captured_cfg['frozen'] is True
    assert list(feature_shape) == [1, 1, 4, 4]
    torch.testing.assert_close(features, torch.full((1, 16), 7.0))


def test_build_memory_bank_fits_pca_and_enables_predict(monkeypatch):
    import baoiad.models.backbone_utils as backbone_utils

    monkeypatch.setattr(backbone_utils, 'build_feature_extractor', lambda *args, **kwargs: _DummyFeatureExtractor())

    model = DFMDetector(backbone='resnet18', layer='layer3', pooling_kernel_size=2, pca_level=0.9)
    data_samples = _make_data_samples(2, 8, 8)

    model.eval()
    try:
        model(torch.randn(2, 3, 8, 8), data_samples, mode='predict')
    except RuntimeError as exc:
        assert 'build_memory_bank()/fit()' in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError('predict() should fail before PCA fitting.')

    model.train()
    for offset in range(3):
        inputs = torch.randn(2, 3, 8, 8) + float(offset)
        out = model(inputs, _make_data_samples(2, 8, 8), mode='loss')
        assert isinstance(out, dict)

    assert len(model._memory_bank) == 3
    model.build_memory_bank()

    assert len(model._memory_bank) == 0
    assert model.pca_model.mean.numel() > 0
    assert model.pca_model.singular_vectors.numel() > 0

    model.eval()
    results = model(torch.randn(2, 3, 8, 8), data_samples, mode='predict')

    assert isinstance(results, list)
    assert len(results) == 2
    assert all(torch.isfinite(torch.tensor(result.pred_score)) for result in results)

    maps = torch.stack([result.pred_anomaly_map for result in results])
    assert list(maps.shape) == [2, 1, 8, 8]
    assert torch.isfinite(maps).all()


def test_single_sample_memory_bank_fit_does_not_raise_index_error(monkeypatch):
    import baoiad.models.backbone_utils as backbone_utils

    monkeypatch.setattr(backbone_utils, 'build_feature_extractor', lambda *args, **kwargs: _DummyFeatureExtractor())

    model = DFMDetector(backbone='resnet18', layer='layer3', pooling_kernel_size=2, pca_level=0.97)
    sample = _make_data_samples(1, 8, 8)

    model.train()
    model(torch.randn(1, 3, 8, 8), sample, mode='loss')
    model.build_memory_bank()

    model.eval()
    results = model(torch.randn(1, 3, 8, 8), sample, mode='predict')

    assert len(results) == 1
    assert torch.isfinite(torch.tensor(results[0].pred_score))
    assert torch.isfinite(results[0].pred_anomaly_map).all()



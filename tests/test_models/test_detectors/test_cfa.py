"""Tests for CFADetector."""

import torch
import pytest

import baoiad  # noqa: F401
from baoiad.registry import MODELS
from baoiad.structures import ADDataSample


def _make_cfg(**overrides):
    cfg = dict(
        type='CFADetector',
        backbone=dict(
            type='FeatureExtractor',
            backbone_name='resnet18',
            pretrained=False,
            out_indices=(1, 2, 3),
            frozen=True,
        ),
        gamma_c=1,
        gamma_d=1,
        num_nearest_neighbors=3,
        num_hard_negative_features=3,
        radius=1e-5,
        num_init_batches=2,
        sigma=4,
    )
    cfg.update(overrides)
    return cfg


def _make_data_samples(batch_size, height=64, width=64):
    samples = []
    for i in range(batch_size):
        sample = ADDataSample()
        sample.gt_label = i % 2
        sample.gt_mask = torch.zeros(height, width)
        sample.set_metainfo({
            'cls_name': 'bottle',
            'img_path': f'/fake/{i}.png',
            'defect_type': 'good',
        })
        samples.append(sample)
    return samples


def _make_batch(batch_size=2, height=64, width=64):
    return torch.randn(batch_size, 3, height, width), _make_data_samples(batch_size, height, width)


def test_descriptor_matches_expected_backbone_channels():
    model = MODELS.build(_make_cfg())

    assert model.backbone.out_channels == (64, 128, 256)
    assert model.descriptor.layer.conv.in_channels == sum(model.backbone.out_channels) + 2

    inputs, _ = _make_batch()
    features = model(inputs, mode='tensor')
    assert tuple(features.shape[:2]) == (2, 448)


def test_forward_loss_initializes_memory_bank_after_required_batches():
    model = MODELS.build(_make_cfg(num_init_batches=2))
    model.train()

    inputs, data_samples = _make_batch()
    first = model(inputs, data_samples, mode='loss')
    assert isinstance(first, dict)
    assert first['loss'].item() == 0.0
    assert model.memory_bank.numel() == 0

    second = model(inputs, data_samples, mode='loss')
    assert isinstance(second, dict)
    assert second['loss'].item() == 0.0
    assert model.memory_bank.numel() > 0


def test_predict_requires_initialized_memory_bank():
    model = MODELS.build(_make_cfg())
    model.eval()
    inputs, data_samples = _make_batch()

    with pytest.raises(RuntimeError, match='memory bank'):
        model(inputs, data_samples, mode='predict')


def test_build_memory_bank_without_loader_finalizes_collected_features():
    model = MODELS.build(_make_cfg(num_init_batches=3))
    model.train()
    inputs, data_samples = _make_batch()

    model(inputs, data_samples, mode='loss')
    assert model.memory_bank.numel() == 0

    model.build_memory_bank()
    assert model.memory_bank.numel() > 0


def test_build_memory_bank_with_loader_enables_non_zero_predict():
    model = MODELS.build(_make_cfg(num_init_batches=2))
    model.eval()

    warmup_loader = [
        {
            'inputs': torch.randn(2, 3, 64, 64),
            'data_samples': _make_data_samples(2, 64, 64),
        }
        for _ in range(2)
    ]

    model.build_memory_bank(warmup_loader)
    assert model.memory_bank.numel() > 0

    inputs, data_samples = _make_batch()
    outputs = model(inputs, data_samples, mode='predict')

    assert isinstance(outputs, list)
    assert len(outputs) == 2
    scores = torch.tensor([output.pred_score for output in outputs], dtype=torch.float32)
    maps = torch.stack([output.pred_anomaly_map for output in outputs])
    assert torch.isfinite(scores).all()
    assert torch.isfinite(maps).all()
    assert maps.abs().sum().item() > 0

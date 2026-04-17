"""Tests for CSFlowDetector."""

import math

import pytest
import torch
import torch.nn.functional as F

from baoiad.structures import ADDataSample

import baoiad  # noqa: F401
from baoiad.registry import MODELS
from baoiad.models.detectors.csflow import ParallelGlowCouplingLayer


@MODELS.register_module()
class _ToyCSFlowBackbone(torch.nn.Module):
    """Cheap test double that preserves CSFlow's expected feature geometry."""

    def __init__(self, channels=304):
        super().__init__()
        self.channels = channels

    def forward(self, inputs):
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)
        base = inputs.mean(dim=1, keepdim=True)
        features = []
        for size in ((8, 8), (4, 4), (2, 2)):
            pooled = F.adaptive_avg_pool2d(base, size)
            features.append(pooled.repeat(1, self.channels, 1, 1))
        return features

    def train(self, mode=True):
        return super().train(False)


def _make_data_samples(batch_size, height=256, width=256):
    samples = []
    for index in range(batch_size):
        sample = ADDataSample()
        sample.gt_label = index % 2
        sample.gt_mask = torch.zeros(height, width)
        sample.set_metainfo({
            'cls_name': 'bottle',
            'img_path': f'/fake/{index}.png',
            'defect_type': 'good',
        })
        samples.append(sample)
    return samples


def _build_model():
    return MODELS.build(dict(
        type='CSFlowDetector',
        input_size=(256, 256),
        n_coupling_blocks=1,
        cross_conv_hidden_channels=32,
        clamp=3,
        backbone=dict(type='_ToyCSFlowBackbone'),
    ))


@pytest.fixture(scope='module')
def model():
    return _build_model()


def test_tensor_mode_returns_three_feature_scales(model):
    model.eval()

    with torch.no_grad():
        outputs = model(torch.randn(2, 3, 256, 256), mode='tensor')

    assert isinstance(outputs, list)
    assert len(outputs) == 3
    assert [tuple(tensor.shape) for tensor in outputs] == [
        (2, 304, 8, 8),
        (2, 304, 4, 4),
        (2, 304, 2, 2),
    ]
    assert all(torch.isfinite(tensor).all().item() for tensor in outputs)


def test_loss_matches_reference_formula_and_is_finite(model):
    model.train()
    inputs = torch.randn(2, 3, 256, 256)
    data_samples = _make_data_samples(2)

    features = model.feature_extractor(inputs)
    z_dist, jacobians = model.graph(features)
    concatenated = torch.cat([latent.reshape(latent.shape[0], -1) for latent in z_dist], dim=1)
    expected = torch.mean(0.5 * torch.sum(concatenated ** 2, dim=1) - jacobians) / concatenated.shape[1]

    losses = model(inputs, data_samples, mode='loss')

    assert isinstance(losses, dict)
    assert 'loss' in losses
    assert torch.isfinite(losses['loss']).item()
    assert torch.allclose(losses['loss'], expected, rtol=1e-5, atol=1e-6)


def test_predict_outputs_scores_and_maps_with_expected_shapes(model):
    model.eval()
    data_samples = _make_data_samples(2)

    with torch.no_grad():
        predictions = model(torch.randn(2, 3, 256, 256), data_samples, mode='predict')

    assert isinstance(predictions, list)
    assert len(predictions) == 2

    for sample in predictions:
        assert math.isfinite(sample.pred_score)
        assert sample.pred_anomaly_map.shape == (1, 256, 256)
        assert torch.isfinite(sample.pred_anomaly_map).all().item()
        assert torch.all(sample.pred_anomaly_map >= 0).item()


def test_train_keeps_feature_extractor_frozen(model):
    model.train()

    assert model.feature_extractor.training is False
    assert not any(parameter.requires_grad for parameter in model.feature_extractor.parameters())


def test_parallel_glow_coupling_layer_is_self_invertible():
    torch.manual_seed(0)
    layer = ParallelGlowCouplingLayer(
        dims_in=[(8, 8, 8), (8, 4, 4), (8, 2, 2)],
        subnet_args=dict(channels_hidden=16, kernel_size=3),
        clamp=3,
    )
    inputs = [
        torch.randn(2, 8, 8, 8),
        torch.randn(2, 8, 4, 4),
        torch.randn(2, 8, 2, 2),
    ]

    with torch.no_grad():
        outputs, _ = layer(inputs, rev=False)
        reconstructed, _ = layer(outputs, rev=True)

    for original, recovered in zip(inputs, reconstructed):
        assert torch.allclose(original, recovered, atol=1e-5, rtol=1e-5)


def test_parallel_glow_coupling_layer_reverse_matches_reference_snapshot_formula():
    torch.manual_seed(0)
    layer = ParallelGlowCouplingLayer(
        dims_in=[(8, 8, 8), (8, 4, 4), (8, 2, 2)],
        subnet_args=dict(channels_hidden=16, kernel_size=3),
        clamp=3,
    )
    inputs = [
        torch.randn(2, 8, 8, 8),
        torch.randn(2, 8, 4, 4),
        torch.randn(2, 8, 2, 2),
    ]

    with torch.no_grad():
        outputs, _ = layer(inputs, rev=False)
        reversed_outputs, _ = layer(outputs, rev=True)

        x01, x02 = outputs[0].narrow(1, 0, layer.split_len1), outputs[0].narrow(1, layer.split_len1, layer.split_len2)
        x11, x12 = outputs[1].narrow(1, 0, layer.split_len1), outputs[1].narrow(1, layer.split_len1, layer.split_len2)
        x21, x22 = outputs[2].narrow(1, 0, layer.split_len1), outputs[2].narrow(1, layer.split_len1, layer.split_len2)

        r01, r11, r21 = layer.cross_convolution1(x01, x11, x21)
        s01, t01 = r01[:, :layer.split_len2], r01[:, layer.split_len2:]
        s11, t11 = r11[:, :layer.split_len2], r11[:, layer.split_len2:]
        s21, t21 = r21[:, :layer.split_len2], r21[:, layer.split_len2:]

        y02 = (x02 - t01) / layer._exp(s01)
        y12 = (x12 - t11) / layer._exp(s11)
        y22 = (x22 - t21) / layer._exp(s21)

        r02, r12, r22 = layer.cross_convolution2(y02, y12, y22)
        s02, t02 = r02[:, :layer.split_len2], r01[:, layer.split_len2:]
        s12, t12 = r12[:, :layer.split_len2], r11[:, layer.split_len2:]
        s22, t22 = r22[:, :layer.split_len2], r21[:, layer.split_len2:]

        expected = [
            torch.cat(((x01 - t02) / layer._exp(s02), y02), dim=1),
            torch.cat(((x11 - t12) / layer._exp(s12), y12), dim=1),
            torch.cat(((x21 - t22) / layer._exp(s22), y22), dim=1),
        ]

    for actual, reference in zip(reversed_outputs, expected):
        assert torch.allclose(actual, reference, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    ('image_score_mode', 'expected_kind'),
    [
        ('scale0_mean', 'scale0_mean'),
        ('scale0_map_max', 'scale0_map_max'),
    ],
)
def test_image_score_modes_match_manual_formula(image_score_mode, expected_kind):
    model = MODELS.build(dict(
        type='CSFlowDetector',
        input_size=(256, 256),
        n_coupling_blocks=1,
        cross_conv_hidden_channels=32,
        clamp=3,
        image_score_mode=image_score_mode,
        backbone=dict(type='_ToyCSFlowBackbone'),
    ))
    model.eval()
    data_samples = _make_data_samples(2)
    inputs = torch.randn(2, 3, 256, 256)

    with torch.no_grad():
        features = model.feature_extractor(inputs)
        z_dist, _ = model.graph(features)
        results = model(inputs, data_samples, mode='predict')

    if expected_kind == 'scale0_mean':
        scale0 = z_dist[0].reshape(z_dist[0].shape[0], -1)
        expected = torch.mean(scale0 ** 2 / 2, dim=1)
    else:
        scale0_map = model._scale_maps(z_dist)[0]
        expected = scale0_map.flatten(1).max(dim=1).values

    actual = torch.tensor([float(sample.pred_score) for sample in results], dtype=torch.float32)
    assert torch.allclose(actual, expected.cpu(), atol=1e-6, rtol=1e-6)

"""Tests for DifferNetDetector."""

import math

import torch

import baoiad  # noqa: F401
from baoiad.models.detectors.differnet import SubnetFC
from baoiad.registry import MODELS
from baoiad.structures import ADDataSample


def _make_data_samples(batch_size, height=64, width=64):
    samples = []
    for i in range(batch_size):
        sample = ADDataSample()
        sample.gt_label = i % 2
        sample.gt_mask = torch.zeros(height, width)
        sample.cls_name = 'bottle'
        sample.img_path = f'/fake/{i}.png'
        sample.defect_type = 'good'
        samples.append(sample)
    return samples


def _build_model(**overrides):
    cfg = dict(
        type='DifferNetDetector',
        backbone='alexnet',
        n_coupling_blocks=2,
        clamp=3.0,
        pretrained=False,
        n_transforms=2,
        n_train_transforms=2,
        scales=((128, 128), (96, 96), (64, 64)),
    )
    cfg.update(overrides)
    return MODELS.build(cfg)


def test_forward_tensor_returns_expected_feature_shape():
    model = _build_model()
    model.eval()

    features = model(torch.randn(2, 3, 64, 64), mode='tensor')

    assert tuple(features.shape) == (2, 768)
    assert torch.isfinite(features).all()


def test_forward_loss_returns_finite_scalar():
    model = _build_model()
    model.train()

    losses = model(torch.randn(2, 3, 64, 64), _make_data_samples(2), mode='loss')

    assert set(losses.keys()) == {'loss'}
    assert losses['loss'].ndim == 0
    assert torch.isfinite(losses['loss'])


def test_forward_predict_returns_uniform_maps_matching_scores():
    model = _build_model()
    model.eval()

    results = model(torch.randn(2, 3, 64, 64), _make_data_samples(2), mode='predict')

    assert len(results) == 2
    for sample in results:
        assert hasattr(sample, 'pred_score')
        assert hasattr(sample, 'pred_anomaly_map')
        assert math.isfinite(float(sample.pred_score))
        assert tuple(sample.pred_anomaly_map.shape) == (1, 64, 64)
        assert torch.isfinite(sample.pred_anomaly_map).all()
        expected_map = torch.full((1, 64, 64), float(sample.pred_score))
        assert torch.allclose(sample.pred_anomaly_map, expected_map)


def test_subnet_fc_uses_reference_internal_size():
    model = _build_model()
    coupling = model.flow.layers[1]

    assert coupling.s1.fc1.in_features == 384
    assert coupling.s1.fc1.out_features == 2048
    assert coupling.s1.fc3.out_features == 768
    assert coupling.s2.fc1.in_features == 384
    assert coupling.s2.fc1.out_features == 2048
    assert coupling.s2.fc3.out_features == 768


def test_subnet_fc_internal_size_override():
    model = _build_model(fc_internal_size=1536)
    coupling = model.flow.layers[1]

    assert coupling.s1.fc1.out_features == 1536
    assert coupling.s2.fc1.out_features == 1536


def test_subnet_fc_forward_skips_batchnorm(monkeypatch):
    subnet = SubnetFC(size_in=4, size=6)
    called = False

    def _raise_if_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError('BatchNorm should not be used in SubnetFC.forward().')

    monkeypatch.setattr(subnet.bn, 'forward', _raise_if_called)

    out = subnet(torch.randn(2, 4))

    assert tuple(out.shape) == (2, 6)
    assert called is False


def test_loss_rotates_each_image_individually(monkeypatch):
    model = _build_model(n_train_transforms=2, n_transforms=1)
    recorded_angles = []
    angle_iter = iter([-10.0, 10.0, -20.0, 20.0])

    monkeypatch.setattr(model, '_apply_color_jitter', lambda x: x)

    def fake_rotate(tensor, angle):
        recorded_angles.append(angle)
        return tensor

    def fake_extract_features(x):
        return torch.zeros(x.shape[0], 768, device=x.device)

    def fake_flow_forward(feats, rev=False):
        return feats, torch.zeros(feats.shape[0], device=feats.device)

    monkeypatch.setattr(model, '_rotate_tensor', fake_rotate)
    monkeypatch.setattr(model, 'extract_features', fake_extract_features)
    monkeypatch.setattr(model.flow, 'forward', fake_flow_forward)
    monkeypatch.setattr(
        'baoiad.models.detectors.differnet.random.uniform',
        lambda _low, _high: next(angle_iter),
    )

    losses = model(torch.randn(2, 3, 64, 64), _make_data_samples(2), mode='loss')

    assert recorded_angles == [-10.0, 10.0, -20.0, 20.0]
    assert torch.isfinite(losses['loss'])


def test_loss_matches_reference_dimension_normalized_nll(monkeypatch):
    model = _build_model(n_train_transforms=2, n_transforms=1)

    monkeypatch.setattr(model, '_apply_color_jitter', lambda x: x)
    monkeypatch.setattr(model, '_apply_random_rotations', lambda x: x)
    monkeypatch.setattr(
        model,
        'extract_features',
        lambda x: torch.ones(x.shape[0], 768, device=x.device),
    )
    monkeypatch.setattr(
        model.flow,
        'forward',
        lambda feats, rev=False: (feats, torch.zeros(feats.shape[0], device=feats.device)),
    )

    losses = model(torch.randn(2, 3, 64, 64), _make_data_samples(2), mode='loss')

    assert losses['loss'] == torch.tensor(0.5)


def test_predict_rotates_each_image_individually(monkeypatch):
    model = _build_model(n_transforms=2, n_train_transforms=1)
    recorded_angles = []
    angle_iter = iter([-15.0, 15.0, -30.0, 30.0])

    monkeypatch.setattr(model, '_apply_color_jitter', lambda x: x)

    def fake_rotate(tensor, angle):
        recorded_angles.append(angle)
        return tensor

    monkeypatch.setattr(model, '_rotate_tensor', fake_rotate)
    monkeypatch.setattr(
        model,
        'extract_features',
        lambda x: torch.zeros(x.shape[0], 768, device=x.device),
    )
    monkeypatch.setattr(
        model.flow,
        'forward',
        lambda feats, rev=False: (feats, torch.zeros(feats.shape[0], device=feats.device)),
    )
    monkeypatch.setattr(
        'baoiad.models.detectors.differnet.random.uniform',
        lambda _low, _high: next(angle_iter),
    )

    results = model(torch.randn(2, 3, 64, 64), _make_data_samples(2), mode='predict')

    assert len(results) == 2
    assert recorded_angles == [-15.0, 15.0, -30.0, 30.0]


def test_loss_can_disable_dimension_normalization(monkeypatch):
    model = _build_model(
        n_train_transforms=2,
        n_transforms=1,
        loss_normalize_by_dim=False,
    )

    monkeypatch.setattr(model, '_apply_color_jitter', lambda x: x)
    monkeypatch.setattr(model, '_apply_random_rotations', lambda x: x)
    monkeypatch.setattr(
        model,
        'extract_features',
        lambda x: torch.ones(x.shape[0], 768, device=x.device),
    )
    monkeypatch.setattr(
        model.flow,
        'forward',
        lambda feats, rev=False: (feats, torch.zeros(feats.shape[0], device=feats.device)),
    )

    losses = model(torch.randn(2, 3, 64, 64), _make_data_samples(2), mode='loss')

    assert losses['loss'] == torch.tensor(384.0)


def test_predict_can_use_fixed_test_rotations(monkeypatch):
    model = _build_model(n_transforms=4, test_rotation_mode='fixed')
    recorded_angles = []

    monkeypatch.setattr(model, '_apply_color_jitter', lambda x: x)

    def fake_rotate(tensor, angle):
        recorded_angles.append(angle)
        return tensor

    monkeypatch.setattr(model, '_rotate_tensor', fake_rotate)
    monkeypatch.setattr(
        model,
        'extract_features',
        lambda x: torch.zeros(x.shape[0], 768, device=x.device),
    )
    monkeypatch.setattr(
        model.flow,
        'forward',
        lambda feats, rev=False: (feats, torch.zeros(feats.shape[0], device=feats.device)),
    )

    results = model(torch.randn(2, 3, 64, 64), _make_data_samples(2), mode='predict')

    assert len(results) == 2
    assert recorded_angles == [0.0, 90.0, 180.0, 270.0]

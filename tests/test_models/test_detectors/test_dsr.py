"""Tests for DSRDetector."""

from __future__ import annotations

import warnings
from pathlib import Path
from unittest import TestCase

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import baoiad  # noqa: F401
from baoiad.checkpoint import CheckpointLoadError
from baoiad.models.detectors.dsr import (
    DSRDetector,
    DiscreteLatentModel,
    _generate_perlin_anomaly_batch,
)
from baoiad.registry import MODELS
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


def _make_inputs(batch_size, H=64, W=64):
    return torch.rand(batch_size, 3, H, W)


def _build_model(**overrides):
    cfg = dict(
        type='DSRDetector',
        embedding_dim=32,
        num_embeddings=64,
        num_hiddens=32,
        num_residual_layers=1,
        num_residual_hiddens=16,
        phase=2,
        pretrained_vqvae_path=None,
    )
    cfg.update(overrides)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        return MODELS.build(cfg)


class _IdentityPhase3Augmenter(nn.Module):
    def __init__(self):
        super().__init__()
        self.transforms = []

    def __call__(self, image):
        return image


class _FakeDiscreteLatentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vq_vae_top = nn.Identity()
        self.vq_vae_bot = nn.Identity()
        self.upsample_t = nn.Identity()

    def forward(self, inputs):
        batch_size, _, height, width = inputs.shape
        zeros = torch.zeros(batch_size, 1, height, width, device=inputs.device)
        return {
            'recon_image': torch.zeros_like(inputs),
            'quantized_t': zeros,
            'quantized_b': zeros,
        }


class _FakeRestrictionModule(nn.Module):
    def forward(self, embeddings, _quantization):
        return embeddings, embeddings


class _FakeImageReconstructionNetwork(nn.Module):
    def forward(self, embeddings):
        return torch.zeros(embeddings.shape[0], 3, embeddings.shape[2], embeddings.shape[3], device=embeddings.device)


class _FixedMaskModule(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer('logits', logits)

    def forward(self, *args):
        batch_size = args[0].shape[0]
        return self.logits.expand(batch_size, -1, -1, -1)


class TestDSRDetector(TestCase):
    def test_forward_tensor(self):
        model = _build_model()
        model.eval()
        out = model(_make_inputs(1), mode='tensor')
        assert out is not None
        assert sorted(out.keys()) == ['quantized_b', 'quantized_t', 'recon_image']

    def test_forward_loss_phase2_is_finite(self):
        model = _build_model(phase=2)
        model.train()
        data_samples = _make_data_samples(1, 64, 64)
        out = model(_make_inputs(1), data_samples, mode='loss')
        assert isinstance(out, dict)
        assert 'loss' in out
        assert torch.isfinite(out['loss'])

    def test_forward_loss_phase3_is_finite(self):
        model = _build_model(phase=3)
        model.phase3_augmenters = _IdentityPhase3Augmenter()
        model.train()
        data_samples = _make_data_samples(1, 64, 64)
        out = model(_make_inputs(1), data_samples, mode='loss')
        assert isinstance(out, dict)
        assert 'loss' in out
        assert torch.isfinite(out['loss'])

    def test_forward_predict_outputs_finite_scores_and_maps(self):
        model = _build_model()
        model.eval()
        data_samples = _make_data_samples(1, 64, 64)
        out = model(_make_inputs(1), data_samples, mode='predict')
        assert isinstance(out, list)
        assert len(out) == 1
        for sample in out:
            assert hasattr(sample, 'pred_score')
            assert hasattr(sample, 'pred_anomaly_map')
            assert torch.isfinite(torch.tensor(sample.pred_score))
            assert tuple(sample.pred_anomaly_map.shape) == (1, 64, 64)
            assert torch.isfinite(sample.pred_anomaly_map).all()
            assert torch.all((sample.pred_anomaly_map >= 0) & (sample.pred_anomaly_map <= 1))

    def test_set_epoch_info_switches_to_phase3_and_freezes_non_upsampling_params(self):
        model = _build_model(phase=2, upsampling_train_ratio=0.7)
        assert any(param.requires_grad for param in model.anomaly_detection_module.parameters())
        assert all(param.requires_grad for param in model.upsampling_module.parameters())

        model.set_epoch_info(epoch=70, max_epochs=100)

        assert model.phase == 3
        non_upsampling_trainable = [
            name for name, param in model.named_parameters()
            if param.requires_grad and 'upsampling_module' not in name
        ]
        upsampling_trainable = [
            name for name, param in model.named_parameters()
            if param.requires_grad and 'upsampling_module' in name
        ]
        assert non_upsampling_trainable == []
        assert upsampling_trainable

    def test_phase3_augmenter_matches_reference_shape(self):
        model = _build_model()
        transform_types = {type(transform).__name__ for transform in model.phase3_augmenters.transforms}
        assert model.phase3_augmenters.num_transforms == 3
        assert {'RandomSolarize', 'AutoAugment', 'RandomAffine'} <= transform_types

    def test_predict_uses_pre_upsampling_scores_and_post_upsampling_maps(self):
        model = _build_model()
        model.discrete_latent_model = _FakeDiscreteLatentModel()
        model.subspace_restriction_module_hi = _FakeRestrictionModule()
        model.subspace_restriction_module_lo = _FakeRestrictionModule()
        model.image_reconstruction_network = _FakeImageReconstructionNetwork()

        pre_logits = torch.zeros(1, 2, 8, 8)
        pre_logits[:, 1, 4, 4] = 10.0
        up_logits = torch.zeros(1, 2, 8, 8)
        up_logits[:, 1, :, :] = -10.0

        model.anomaly_detection_module = _FixedMaskModule(pre_logits)
        model.upsampling_module = _FixedMaskModule(up_logits)

        anomaly_map, img_scores = model._forward_predict(_make_inputs(1, 8, 8))
        expected_score = torch.amax(
            F.avg_pool2d(torch.softmax(pre_logits, dim=1)[:, 1:, :, :], 21, stride=1, padding=10),
            dim=(2, 3),
        ).squeeze(1)
        expected_map = torch.softmax(up_logits, dim=1)[:, 1:, :, :]

        assert torch.allclose(img_scores, expected_score)
        assert torch.allclose(anomaly_map, expected_map)


def test_generate_perlin_anomaly_batch_returns_binary_masks():
    masks = _generate_perlin_anomaly_batch(batch_size=4, H=32, W=32)
    assert tuple(masks.shape) == (4, 1, 32, 32)
    assert torch.isfinite(masks).all()
    assert set(torch.unique(masks).tolist()) <= {0.0, 1.0}


def test_resolve_vqvae_path_prefers_local_pckl(monkeypatch, tmp_path):
    candidate = tmp_path / 'pre_trained' / 'vq_model_pretrained_128_4096.pckl'
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(b'checkpoint')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        'baoiad.models.detectors.dsr._download_vqvae_weights',
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('download should not be used')),
    )

    resolved = DSRDetector._resolve_vqvae_path('auto')

    assert Path(resolved).resolve() == candidate


def test_resolve_vqvae_path_accepts_pretrained_alias(tmp_path):
    candidate = tmp_path / 'pre_trained' / 'vq_model_pretrained_128_4096.pckl'
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(b'checkpoint')

    resolved = DSRDetector._resolve_vqvae_path(str(tmp_path / 'pretrained' / 'vq_model_pretrained_128_4096.pth'))

    assert resolved == str(candidate)


def test_vqvae_compatibility_check_distinguishes_reference_and_legacy_keys():
    compatible = DiscreteLatentModel(
        num_hiddens=32,
        num_residual_layers=1,
        num_residual_hiddens=16,
        num_embeddings=64,
        embedding_dim=32,
    ).state_dict()
    incompatible = {'conv1.weight': torch.randn(1)}

    assert DSRDetector._is_compatible_vqvae_checkpoint(compatible)
    assert not DSRDetector._is_compatible_vqvae_checkpoint(incompatible)


def test_auto_pretrained_loading_retries_download_for_incompatible_local_checkpoint(monkeypatch):
    compatible = DiscreteLatentModel(
        num_hiddens=32,
        num_residual_layers=1,
        num_residual_hiddens=16,
        num_embeddings=64,
        embedding_dim=32,
    ).state_dict()
    load_calls = []

    def fake_load(path, *args, **kwargs):
        load_calls.append(path)
        if path == 'bad_local.pckl':
            return {'conv1.weight': torch.randn(1)}
        if path == 'downloaded_good.pckl':
            return compatible
        raise AssertionError(f'unexpected checkpoint path: {path}')

    monkeypatch.setattr(DSRDetector, '_resolve_vqvae_path', staticmethod(lambda _: 'bad_local.pckl'))
    monkeypatch.setattr('baoiad.models.detectors.dsr._download_vqvae_weights', lambda **kwargs: 'downloaded_good.pckl')
    monkeypatch.setattr(
        'baoiad.models.detectors.dsr.load_baoiad_checkpoint', fake_load)

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        _build_model(pretrained_vqvae_path='auto')

    assert load_calls == ['bad_local.pckl', 'downloaded_good.pckl']


def test_auto_pretrained_loading_does_not_swallow_checkpoint_errors(monkeypatch):
    expected = CheckpointLoadError('checkpoint is corrupt')
    download_called = False

    def fail_load(_path):
        raise expected

    def unexpected_download(**_kwargs):
        nonlocal download_called
        download_called = True
        return 'unused.pckl'

    detector = object.__new__(DSRDetector)
    monkeypatch.setattr(detector, '_load_vqvae_checkpoint', fail_load)
    monkeypatch.setattr(
        'baoiad.models.detectors.dsr._download_vqvae_weights',
        unexpected_download,
    )

    with pytest.raises(CheckpointLoadError) as exc_info:
        detector._try_load_discrete_latent_model_weights(
            'corrupt.pckl', allow_auto_redownload=True)

    assert exc_info.value is expected
    assert not download_called

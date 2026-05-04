"""Tests for PyramidFlowDetector (Faithful Implementation)."""

from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from baoiad.structures import ADDataSample

import baoiad  # noqa: F401
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


class TestPyramidFlowDetector(TestCase):
    """Tests for the faithful PyramidFlow implementation."""

    def setUp(self):
        # Use small model for faster testing
        self.cfg = dict(
            type='PyramidFlowDetector',
            encoder='resnet18',
            channel=64,
            num_level=4,
            num_stack=2,  # Reduced for testing
            ksize=7,
            vn_dims=(0, 1),
            save_memory=False,
        )

    def test_forward_tensor(self):
        """Test tensor mode returns encoder features."""
        model = MODELS.build(self.cfg)
        model.eval()
        # Use size divisible by 8 for pyramid levels (256 -> 64 after encoder, then 4 pyramid levels)
        out = model(torch.randn(2, 3, 256, 256), mode='tensor')
        assert out is not None
        assert out.shape[0] == 2

    def test_forward_loss(self):
        """Test loss mode computes FFT loss on batch differences."""
        model = MODELS.build(self.cfg)
        model.train()
        data_samples = _make_data_samples(2, 256, 256)
        out = model(torch.randn(2, 3, 256, 256), data_samples, mode='loss')
        assert isinstance(out, dict)
        assert 'loss' in out
        assert out['loss'].requires_grad

    def test_forward_predict_without_template(self):
        """Test predict mode raises error without template."""
        model = MODELS.build(self.cfg)
        model.eval()
        data_samples = _make_data_samples(2, 256, 256)

        with pytest.raises(RuntimeError, match="Template not built"):
            model(torch.randn(2, 3, 256, 256), data_samples, mode='predict')

    def test_forward_predict_with_template(self):
        """Test predict mode with manually set template."""
        model = MODELS.build(self.cfg)
        model.eval()

        # Create fake template (4 pyramid levels)
        template = tuple(
            torch.randn(1, 64, 64 // (2 ** i), 64 // (2 ** i))
            for i in range(4)
        )
        model.template = template
        model._template_built = True

        data_samples = _make_data_samples(2, 256, 256)
        out = model(torch.randn(2, 3, 256, 256), data_samples, mode='predict')

        assert isinstance(out, list)
        assert len(out) == 2
        assert hasattr(out[0], 'pred_score')
        assert hasattr(out[0], 'pred_score_mean')
        assert hasattr(out[0], 'pred_score_max')
        assert hasattr(out[0], 'pred_anomaly_map')

    def test_forward_predict_without_resize_keeps_feature_scale(self):
        model = MODELS.build({**self.cfg, 'predict_resize_to_input': False})
        model.eval()
        template = tuple(
            torch.randn(1, 64, 64 // (2 ** i), 64 // (2 ** i))
            for i in range(4)
        )
        model.template = template
        model._template_built = True

        data_samples = _make_data_samples(1, 256, 256)
        out = model(torch.randn(1, 3, 256, 256), data_samples, mode='predict')
        assert out[0].pred_anomaly_map.shape == (1, 64, 64)

    def test_forward_predict_can_drop_selected_levels(self):
        model = MODELS.build({
            **self.cfg,
            'encoder': None,
            'num_level': 2,
            'predict_drop_levels': [1],
        })
        model.eval()
        model.template = (
            torch.zeros(1, 1, 1, 1),
            torch.zeros(1, 1, 1, 1),
        )
        model._template_built = True

        def fake_encode_to_latent(imgs):
            del imgs
            return (
                torch.ones(1, 1, 1, 1),
                torch.full((1, 1, 1, 1), 2.0),
            )

        model.core.encode_to_latent = fake_encode_to_latent
        model.core.pyramid.compose_pyramid = lambda levels: sum(levels)

        data_samples = _make_data_samples(1, 1, 1)
        out = model(torch.randn(1, 3, 8, 8), data_samples, mode='predict')

        assert torch.allclose(out[0].pred_anomaly_map, torch.ones(1, 1, 1))
        assert float(out[0].pred_score) == pytest.approx(1.0)

    def test_predict_drop_levels_rejects_invalid_level(self):
        with pytest.raises(ValueError, match='predict_drop_levels'):
            MODELS.build({**self.cfg, 'predict_drop_levels': [9]})

    def test_forward_predict_can_scale_selected_level(self):
        model = MODELS.build({
            **self.cfg,
            'encoder': None,
            'num_level': 2,
            'predict_level_weights': [1.0, 0.5],
        })
        model.eval()
        model.template = (
            torch.zeros(1, 1, 1, 1),
            torch.zeros(1, 1, 1, 1),
        )
        model._template_built = True

        def fake_encode_to_latent(imgs):
            del imgs
            return (
                torch.ones(1, 1, 1, 1),
                torch.full((1, 1, 1, 1), 2.0),
            )

        model.core.encode_to_latent = fake_encode_to_latent
        model.core.pyramid.compose_pyramid = lambda levels: sum(levels)

        data_samples = _make_data_samples(1, 1, 1)
        out = model(torch.randn(1, 3, 8, 8), data_samples, mode='predict')

        assert torch.allclose(out[0].pred_anomaly_map, torch.full((1, 1, 1), 2.0))
        assert float(out[0].pred_score) == pytest.approx(2.0)

    def test_predict_level_weights_rejects_wrong_length(self):
        with pytest.raises(ValueError, match='predict_level_weights'):
            MODELS.build({**self.cfg, 'predict_level_weights': [1.0, 0.5]})

    def test_build_template_uses_single_sample_view_and_batch_average(self):
        model = MODELS.build({
            **self.cfg,
            'encoder': None,
            'num_level': 2,
        })
        model.eval()

        samples = torch.stack([
            torch.full((3, 8, 8), fill_value=float(v))
            for v in (1.0, 2.0, 3.0)
        ])
        dataloader = DataLoader(TensorDataset(samples), batch_size=2, shuffle=True, drop_last=True)

        seen_batch_sizes = []

        def fake_encode_to_latent(imgs):
            seen_batch_sizes.append(int(imgs.shape[0]))
            base = imgs[:, :1, :1, :1].clone()
            return (base, base * 2.0)

        model.core.encode_to_latent = fake_encode_to_latent
        model.build_template_from_dataloader(dataloader, torch.device('cpu'))

        assert seen_batch_sizes == [1, 1, 1]
        assert torch.allclose(model.template[0], torch.tensor([[[[2.0]]]]))
        assert torch.allclose(model.template[1], torch.tensor([[[[4.0]]]]))

    def test_build_template_can_switch_to_eval_pipeline(self):
        model = MODELS.build({
            **self.cfg,
            'encoder': None,
            'num_level': 2,
            'template_pipeline': [],
        })
        model.eval()

        class _Dataset(torch.utils.data.Dataset):
            def __init__(self):
                self.pipeline = lambda data: {'inputs': data['inputs'] + 100.0}
                self.samples = [
                    {'inputs': torch.ones(3, 4, 4)},
                    {'inputs': torch.ones(3, 4, 4) * 2.0},
                ]

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                return self.pipeline(self.samples[idx])

        dataloader = DataLoader(_Dataset(), batch_size=2, shuffle=False, drop_last=True)

        captured = []

        def fake_encode_to_latent(imgs):
            captured.append(float(imgs[0, 0, 0, 0]))
            base = imgs[:, :1, :1, :1].clone()
            return (base,)

        model.core.encode_to_latent = fake_encode_to_latent
        model.build_template_from_dataloader(dataloader, torch.device('cpu'))

        assert captured == [1.0, 2.0]

    def test_encoder_frozen(self):
        """Test that encoder parameters are frozen."""
        model = MODELS.build(self.cfg)
        model.train()

        # Check encoder is frozen
        for param in model.core.inconv.parameters():
            assert not param.requires_grad

    def test_pyramid_decomposition_reconstruction(self):
        """Test that pyramid decomposition and reconstruction is lossless."""
        from baoiad.models.detectors.pyramidflow import LaplacianMaxPyramid

        pyramid_module = LaplacianMaxPyramid(num_levels=4)
        x = torch.randn(1, 64, 64, 64)

        # Build and compose
        pyramid = pyramid_module.build_pyramid(x)
        reconstructed = pyramid_module.compose_pyramid(pyramid)

        assert torch.allclose(x, reconstructed, atol=1e-5)

    def test_pyramid_supports_maxpool_downsample_mode(self):
        from baoiad.models.detectors.pyramidflow import LaplacianMaxPyramid

        pyramid_module = LaplacianMaxPyramid(num_levels=4, downsample_mode='maxpool')
        x = torch.randn(1, 64, 64, 64)
        pyramid = pyramid_module.build_pyramid(x)

        assert len(pyramid) == 4
        assert pyramid[0].shape[-2:] == (64, 64)
        assert pyramid[-1].shape[-2:] == (8, 8)

    def test_pyramid_invalid_downsample_mode_raises(self):
        from baoiad.models.detectors.pyramidflow import LaplacianMaxPyramid

        with pytest.raises(ValueError, match='downsample_mode'):
            LaplacianMaxPyramid(num_levels=4, downsample_mode='bad')

    def test_flow_block_invertibility(self):
        """Test that FlowBlock is approximately invertible."""
        from baoiad.models.detectors.pyramidflow import FlowBlock

        flow_block = FlowBlock(channel=64, direct='up', start_level=0, ksize=7, vn_dims=(0, 1))

        # Create two pyramid levels
        x0 = torch.randn(2, 64, 32, 32)
        x1 = torch.randn(2, 64, 16, 16)
        logdet0 = torch.zeros(2, 64, 32, 32)
        logdet1 = torch.zeros(2, 64, 16, 16)

        inputs = (x0, x1)
        logdets = (logdet0, logdet1)

        # Forward
        outputs, out_logdets = flow_block(inputs, logdets)

        # Inverse
        inputs_recon, logdets_recon = flow_block.inverse(outputs, out_logdets)

        # Check reconstruction
        assert torch.allclose(inputs[0], inputs_recon[0], atol=1e-4)
        assert torch.allclose(inputs[1], inputs_recon[1], atol=1e-4)

    def test_flow_block2_invertibility(self):
        """Test that FlowBlock2 is approximately invertible."""
        from baoiad.models.detectors.pyramidflow import FlowBlock2

        flow_block = FlowBlock2(channel=64, start_level=0, ksize=7, vn_dims=(0, 1))

        # Create three pyramid levels
        x0 = torch.randn(2, 64, 32, 32)
        x1 = torch.randn(2, 64, 16, 16)
        x2 = torch.randn(2, 64, 8, 8)
        logdet0 = torch.zeros(2, 64, 32, 32)
        logdet1 = torch.zeros(2, 64, 16, 16)
        logdet2 = torch.zeros(2, 64, 8, 8)

        inputs = (x0, x1, x2)
        logdets = (logdet0, logdet1, logdet2)

        # Forward
        outputs, out_logdets = flow_block(inputs, logdets)

        # Inverse
        inputs_recon, logdets_recon = flow_block.inverse(outputs, out_logdets)

        # Check reconstruction
        assert torch.allclose(inputs[0], inputs_recon[0], atol=1e-4)
        assert torch.allclose(inputs[1], inputs_recon[1], atol=1e-4)
        assert torch.allclose(inputs[2], inputs_recon[2], atol=1e-4)

    def test_inv_conv_2d_lu(self):
        """Test invertible 1x1 convolution with LU decomposition."""
        from baoiad.models.detectors.pyramidflow import InvConv2dLU

        conv = InvConv2dLU(in_channel=64)
        x = torch.randn(2, 64, 16, 16)

        # Forward
        out = conv(x)

        # Inverse
        x_recon = conv.inverse(out)

        # Check reconstruction
        assert torch.allclose(x, x_recon, atol=1e-4)

    def test_volume_norm_can_reload_spatial_running_mean(self):
        """Checkpoint reload should preserve learned spatial running mean."""
        from baoiad.models.detectors.pyramidflow import VolumeNorm

        norm = VolumeNorm(dims=(0, 1))
        norm.train()
        _ = norm(torch.randn(2, 64, 32, 32))
        state = norm.state_dict()

        reloaded = VolumeNorm(dims=(0, 1))
        reloaded.load_state_dict(state)

        assert reloaded.running_mean.shape == norm.running_mean.shape
        assert torch.allclose(reloaded.running_mean, norm.running_mean)


class TestPyramidFlowCore(TestCase):
    """Tests for PyramidFlowCore module."""

    def test_core_forward_train(self):
        """Test training forward pass."""
        from baoiad.models.detectors.pyramidflow import PyramidFlowCore

        core = PyramidFlowCore(
            encoder='resnet18',
            channel=64,
            num_level=4,
            num_stack=2,
            ksize=7,
            vn_dims=(0, 1),
            batch_size=2,
            save_memory=False
        )
        core.freeze_encoder()

        # Batch size must be 2 for pair training
        imgs = torch.randn(2, 3, 256, 256)
        diff_pixel = core.forward_train(imgs)

        # Output should be (B-1 pairs, H, W) after batch diff
        assert diff_pixel.dim() == 3  # (1, H, W) for 1 pair from batch of 2

    def test_core_encode_predict(self):
        """Test encoding and prediction with template."""
        from baoiad.models.detectors.pyramidflow import PyramidFlowCore

        core = PyramidFlowCore(
            encoder='resnet18',
            channel=64,
            num_level=4,
            num_stack=2,
            ksize=7,
            vn_dims=(0, 1),
            batch_size=2,
            save_memory=False
        )
        core.freeze_encoder()
        core.eval()

        # Encode to latent
        imgs = torch.randn(1, 3, 256, 256)
        pyramid_out = core.encode_to_latent(imgs)

        assert len(pyramid_out) == 4  # 4 pyramid levels

        # Create fake template
        template = tuple(p.mean(dim=0, keepdim=True) for p in pyramid_out)

        # Predict
        anomaly_map = core.predict(imgs, template)
        assert anomaly_map.shape[0] == 1
        assert anomaly_map.shape[1] == 1


def test_build_pyramidflow_resnet18_prefers_local_legacy_checkpoint(tmp_path, monkeypatch):
    from baoiad.models.detectors import pyramidflow as pyramidflow_module

    ckpt_path = tmp_path / 'resnet18-5c106cde.pth'
    torch.save({'conv1.weight': torch.ones(1)}, ckpt_path)

    calls = {}

    class _FakeResNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = torch.nn.BatchNorm2d(64)
            self.relu = torch.nn.ReLU()
            self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = torch.nn.Sequential()

        def load_state_dict(self, state_dict, strict=False):
            calls['state_dict'] = state_dict
            calls['strict'] = strict
            return SimpleNamespace(missing_keys=['bn1.num_batches_tracked'], unexpected_keys=[])

    def _fake_resnet18(*, weights=None, **kwargs):
        calls['weights'] = weights
        return _FakeResNet()

    monkeypatch.setattr(pyramidflow_module, '_LEGACY_RESNET18_PATH', Path(ckpt_path))
    monkeypatch.setattr(pyramidflow_module.models, 'resnet18', _fake_resnet18)

    model = pyramidflow_module._build_pyramidflow_resnet18()

    assert isinstance(model, _FakeResNet)
    assert calls['weights'] is None
    assert calls['strict'] is False
    assert torch.equal(calls['state_dict']['conv1.weight'], torch.ones(1))


def test_build_pyramidflow_resnet18_falls_back_to_torchvision_v1_weights(monkeypatch):
    from baoiad.models.detectors import pyramidflow as pyramidflow_module

    calls = {}

    class _FakeResNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = torch.nn.BatchNorm2d(64)
            self.relu = torch.nn.ReLU()
            self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = torch.nn.Sequential()

    def _fake_resnet18(*, weights=None, **kwargs):
        calls['weights'] = weights
        return _FakeResNet()

    monkeypatch.setattr(pyramidflow_module, '_LEGACY_RESNET18_PATH', Path('/nonexistent/resnet18-5c106cde.pth'))
    monkeypatch.setattr(pyramidflow_module.models, 'resnet18', _fake_resnet18)

    model = pyramidflow_module._build_pyramidflow_resnet18()

    assert isinstance(model, _FakeResNet)
    assert calls['weights'] == pyramidflow_module.models.ResNet18_Weights.IMAGENET1K_V1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

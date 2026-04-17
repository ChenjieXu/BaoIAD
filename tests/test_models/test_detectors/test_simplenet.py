"""Tests for SimpleNetDetector."""

import torch
from unittest import TestCase
from mmengine.optim import OptimWrapper, OptimWrapperDict
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

class TestSimpleNetDetector(TestCase):
    def setUp(self):
        self.cfg = dict(type='SimpleNetDetector', backbone='resnet18', target_dim=128, pretrain_embed_dim=128, dsc_layers=2, dsc_hidden=128, patchsize=1, patchstride=1, pre_proj=1, proj_layer_type=0)
        self.strict_cfg = dict(
            type='SimpleNetDetector',
            strict=True,
            image_size=64,
            gaussian_sigma=4.0,
            backbone='resnet18',
            target_dim=128,
            pretrain_embed_dim=128,
            dsc_layers=2,
            dsc_hidden=128,
            patchsize=1,
            patchstride=1,
            pre_proj=1,
            proj_layer_type=0,
        )

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
        out = model(torch.randn(2, 3, 64, 64), data_samples, mode='predict')
        assert isinstance(out, list)
        assert len(out) == 2

    def test_strict_predict_outputs_full_resolution_map(self):
        model = MODELS.build(self.strict_cfg)
        model.eval()
        data_samples = _make_data_samples(2, 64, 64)
        out = model(torch.randn(2, 3, 64, 64), data_samples, mode='predict')
        assert len(out) == 2
        assert out[0].pred_anomaly_map.shape == (1, 64, 64)
        assert torch.isfinite(out[0].pred_anomaly_map).all()

    def test_strict_train_step_with_multi_optimizer(self):
        model = MODELS.build(self.strict_cfg)
        model.train()
        inputs = torch.randn(2, 3, 64, 64)
        data_samples = _make_data_samples(2, 64, 64)
        optim_wrapper = OptimWrapperDict(
            projection=OptimWrapper(torch.optim.AdamW(model.projection.parameters(), lr=1e-4, weight_decay=1e-2)),
            discriminator=OptimWrapper(torch.optim.Adam(model.discriminator.parameters(), lr=2e-4, weight_decay=1e-5)),
        )

        outputs = model.train_step(dict(inputs=inputs, data_samples=data_samples), optim_wrapper)
        assert 'loss' in outputs
        assert torch.isfinite(outputs['loss'])

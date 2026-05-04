"""Tests for RDPPDetector."""

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


def _make_strict_data_samples(batch_size, H=256, W=256):
    samples = _make_data_samples(batch_size, H, W)
    for sample in samples:
        sample.set_metainfo({'img_noise': torch.randn(3, H, W)})
    return samples


class TestRDPPDetector(TestCase):
    def setUp(self):
        self.cfg = dict(type='RDPPDetector', backbone='wide_resnet50_2')

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

    def test_strict_forward_loss(self):
        model = MODELS.build(dict(type='RDPPDetector', backbone='wide_resnet50_2', strict=True))
        model.train()
        model.proj_loss.sinkhorn = lambda x, y: torch.mean(torch.abs(x - y))
        data_samples = _make_strict_data_samples(2, 64, 64)
        out = model(torch.randn(2, 3, 64, 64), data_samples, mode='loss')
        assert isinstance(out, dict)
        assert torch.isfinite(out['loss'])

    def test_strict_train_step_with_multi_optimizer(self):
        model = MODELS.build(dict(type='RDPPDetector', backbone='wide_resnet50_2', strict=True))
        model.train()
        model.proj_loss.sinkhorn = lambda x, y: torch.mean(torch.abs(x - y))

        inputs = torch.randn(2, 3, 64, 64)
        data_samples = _make_strict_data_samples(2, 64, 64)
        optim_wrapper = OptimWrapperDict(
            projection=OptimWrapper(
                torch.optim.Adam(model.proj_layer.parameters(), lr=1e-3),
                accumulative_counts=2,
            ),
            distillation=OptimWrapper(
                torch.optim.Adam(
                    list(model.ocbe.parameters()) + list(model.student.parameters()),
                    lr=5e-3,
                ),
                accumulative_counts=2,
            ),
        )

        out = model.train_step(dict(inputs=inputs, data_samples=data_samples), optim_wrapper)
        assert 'loss' in out
        assert torch.isfinite(out['loss'])

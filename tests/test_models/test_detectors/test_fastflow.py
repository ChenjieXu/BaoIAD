"""Tests for FastFlowDetector."""

from unittest import TestCase

import torch

import baoiad  # noqa: F401
from baoiad.registry import MODELS
from baoiad.structures import ADDataSample


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


def _flow_kernel_sizes(flow):
    kernel_sizes = []
    for block in flow.module_list:
        conv = block.subnet[1]
        kernel_sizes.append(conv.kernel_size[0])
    return kernel_sizes


class TestFastFlowDetector(TestCase):
    def setUp(self):
        self.cfg = dict(
            type='FastFlowDetector',
            backbone=dict(
                type='TIMMBackbone',
                model_name='resnet18',
                features_only=True,
                out_indices=(1, 2, 3),
                frozen=True,
            ),
            flow_steps=2,
            conv3x3_only=False,
            hidden_ratio=1.0,
            clamp=2.0,
            input_size=(64, 64),
        )

    def test_layer_norm_shapes_match_backbone_feature_shapes(self):
        model = MODELS.build(self.cfg)

        assert len(model.norms) == len(model.backbone.out_channels)
        for norm, channels, reduction in zip(
            model.norms,
            model.backbone.out_channels,
            model.backbone.reduction,
        ):
            expected_shape = [channels, 64 // reduction, 64 // reduction]
            assert list(norm.normalized_shape) == expected_shape
            assert norm.elementwise_affine

    def test_flow_kernel_schedule_matches_reference(self):
        model = MODELS.build(self.cfg)
        assert _flow_kernel_sizes(model.flows[0]) == [3, 1]

        conv3x3_only_cfg = dict(self.cfg, conv3x3_only=True, flow_steps=3)
        conv3x3_only_model = MODELS.build(conv3x3_only_cfg)
        assert _flow_kernel_sizes(conv3x3_only_model.flows[0]) == [3, 3, 3]

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
        assert 'loss' in out
        assert torch.isfinite(out['loss'])

    def test_forward_predict(self):
        model = MODELS.build(self.cfg)
        model.eval()
        data_samples = _make_data_samples(2, 64, 64)
        out = model(torch.randn(2, 3, 64, 64), data_samples, mode='predict')
        assert isinstance(out, list)
        assert len(out) == 2
        for s in out:
            assert hasattr(s, 'pred_score')
            assert hasattr(s, 'pred_anomaly_map')
            assert torch.isfinite(torch.tensor(s.pred_score))
            assert tuple(s.pred_anomaly_map.shape) == (1, 64, 64)
            assert torch.isfinite(s.pred_anomaly_map).all()

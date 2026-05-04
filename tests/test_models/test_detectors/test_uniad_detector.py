"""Tests for UniADDetector."""

import torch
from unittest import TestCase
from baoiad.structures import ADDataSample

import baoiad  # noqa: F401
from baoiad.models.detectors.uniad_detector import UniADDetector
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

class TestUniADDetector(TestCase):
    def setUp(self):
        self.cfg = dict(type='UniADDetector', backbone='wide_resnet50_2', hidden_dim=256, nhead=8, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=512, dropout=0.1, feature_jitter_scale=10.0, neighbor_size=(3, 3))

    def test_forward_tensor(self):
        model = MODELS.build(self.cfg)
        model.eval()
        out = model(torch.randn(2, 3, 256, 256), mode='tensor')
        assert out is not None

    def test_forward_loss(self):
        model = MODELS.build(self.cfg)
        model.train()
        data_samples = _make_data_samples(2, 256, 256)
        out = model(torch.randn(2, 3, 256, 256), data_samples, mode='loss')
        assert isinstance(out, dict)

    def test_forward_predict(self):
        model = MODELS.build(self.cfg)
        model.eval()
        data_samples = _make_data_samples(2, 256, 256)
        out = model(torch.randn(2, 3, 256, 256), data_samples, mode='predict')
        assert isinstance(out, list)
        assert len(out) == 2

    def test_image_score_modes(self):
        score_map = torch.tensor(
            [
                [
                    [10.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 4.0, 4.0],
                    [0.0, 0.0, 4.0, 4.0],
                ]
            ],
            dtype=torch.float32,
        )

        raw_max = UniADDetector._compute_image_scores_from_map(
            score_map,
            mode='raw_max',
            topk=4,
            pool_kernel=2,
        )
        raw_topk_mean = UniADDetector._compute_image_scores_from_map(
            score_map,
            mode='raw_topk_mean',
            topk=4,
            pool_kernel=2,
        )
        pooled_max = UniADDetector._compute_image_scores_from_map(
            score_map,
            mode='pooled_max',
            topk=4,
            pool_kernel=2,
        )
        pooled_topk_mean = UniADDetector._compute_image_scores_from_map(
            score_map,
            mode='pooled_topk_mean',
            topk=4,
            pool_kernel=2,
        )

        assert raw_max.shape == (1,)
        assert raw_topk_mean.shape == (1,)
        assert pooled_max.shape == (1,)
        assert pooled_topk_mean.shape == (1,)
        assert torch.isclose(raw_max[0], torch.tensor(10.0))
        assert torch.isclose(raw_topk_mean[0], torch.tensor(5.5))
        assert torch.isclose(pooled_max[0], torch.tensor(4.0))
        assert pooled_topk_mean[0] <= pooled_max[0]

    def test_invalid_image_score_mode_raises(self):
        with self.assertRaises(ValueError):
            MODELS.build(dict(self.cfg, image_score_mode='bad_mode'))

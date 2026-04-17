"""Tests for ViTADDetector."""

import importlib.util
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from unittest import TestCase
from scipy.ndimage import gaussian_filter
from functools import partial
from torch import nn
from baoiad.structures import ADDataSample

import baoiad  # noqa: F401
from baoiad.registry import MODELS
from baoiad.models.backbones.vitad_backbone import ViTEncoderBackbone
from baoiad.models.detectors.vitad import (
    _vitad_cos_loss,
    _vitad_image_scores,
    _vitad_score_map,
)


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


HAS_TIMM = importlib.util.find_spec('timm') is not None

@pytest.mark.skipif(not HAS_TIMM, reason='timm not installed')
class TestViTADDetector(TestCase):
    def setUp(self):
        self.cfg = dict(type='ViTADDetector', encoder_name='vit_small_patch16_224_dino', img_size=224, decoder_depth=9, pretrained=False)

    def test_forward_tensor(self):
        model = MODELS.build(self.cfg)
        model.eval()
        out = model(torch.randn(2, 3, 224, 224), mode='tensor')
        assert out is not None

    def test_forward_loss(self):
        model = MODELS.build(self.cfg)
        model.train()
        data_samples = _make_data_samples(2, 224, 224)
        out = model(torch.randn(2, 3, 224, 224), data_samples, mode='loss')
        assert isinstance(out, dict)

    def test_forward_predict(self):
        model = MODELS.build(self.cfg)
        model.eval()
        data_samples = _make_data_samples(2, 224, 224)
        out = model(torch.randn(2, 3, 224, 224), data_samples, mode='predict')
        assert isinstance(out, list)
        assert len(out) == 2
        assert out[0].pred_anomaly_map.shape == (1, 224, 224)
        assert torch.isfinite(out[0].pred_anomaly_map).all()
        assert isinstance(float(out[0].pred_score), float)

    def test_reference_cos_loss_matches_official_formula(self):
        torch.manual_seed(0)
        feats_t = [torch.randn(2, 8, 4, 4) for _ in range(3)]
        feats_s = [torch.randn(2, 8, 4, 4) for _ in range(3)]

        actual = _vitad_cos_loss(feats_t, feats_s)
        expected = sum(
            (1 - F.cosine_similarity(
                ft.contiguous().view(ft.shape[0], -1),
                fs.contiguous().view(fs.shape[0], -1),
                dim=1,
            )).mean()
            for ft, fs in zip(feats_t, feats_s)
        )
        assert torch.allclose(actual, expected)

    def test_reference_score_map_matches_official_formula(self):
        torch.manual_seed(0)
        feats_t = [torch.randn(2, 8, 4, 4) for _ in range(3)]
        feats_s = [torch.randn(2, 8, 4, 4) for _ in range(3)]

        actual = _vitad_score_map(feats_t, feats_s, out_size=(32, 32), gaussian_sigma=4.0)

        expected = np.zeros((2, 32, 32), dtype=np.float32)
        for ft, fs in zip(feats_t, feats_s):
            dist = 1 - F.cosine_similarity(ft, fs, dim=1)
            dist = F.interpolate(
                dist.unsqueeze(1),
                size=(32, 32),
                mode='bilinear',
                align_corners=True,
            ).squeeze(1)
            expected += dist.cpu().numpy()
        expected /= (len(feats_t) * len(feats_t))
        for index in range(expected.shape[0]):
            expected[index] = gaussian_filter(expected[index], sigma=4.0)
        assert torch.allclose(actual.cpu(), torch.from_numpy(expected), atol=1e-6, rtol=1e-5)

    def test_reference_image_scores_match_official_pooling(self):
        torch.manual_seed(0)
        score_map = torch.rand(2, 32, 32)
        actual = _vitad_image_scores(score_map)
        expected = F.avg_pool2d(
            score_map.unsqueeze(1),
            kernel_size=16,
            stride=1,
        ).view(score_map.shape[0], -1).max(dim=1).values
        assert torch.allclose(actual, expected)

    def test_timm_metadata_probe_does_not_shift_rng_init(self):
        common = dict(
            teachers=(3, 6, 9),
            neck=(12,),
            img_size=256,
            patch_size=16,
            embed_dim=384,
            depth=12,
            num_heads=6,
            num_classes=0,
            mlp_ratio=4.0,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            pretrained=False,
        )

        torch.manual_seed(123)
        explicit = ViTEncoderBackbone(timm_model=None, **common)
        explicit_weight = explicit.patch_embed.proj.weight.detach().clone()

        torch.manual_seed(123)
        inferred = ViTEncoderBackbone(
            timm_model='vit_small_patch16_224.dino',
            teachers=common['teachers'],
            neck=common['neck'],
            img_size=common['img_size'],
            pretrained=False,
        )
        inferred_weight = inferred.patch_embed.proj.weight.detach().clone()

        assert torch.allclose(explicit_weight, inferred_weight)
        assert 'head.weight' not in inferred.state_dict()

    def test_init_weights_is_noop_for_runner_reentry(self):
        model = MODELS.build(self.cfg)
        before = {
            'net_t.pos_embed': model.state_dict()['net_t.pos_embed'].detach().clone(),
            'net_s.pos_embed': model.state_dict()['net_s.pos_embed'].detach().clone(),
            'net_s.blocks.0.attn.qkv.weight': model.state_dict()['net_s.blocks.0.attn.qkv.weight'].detach().clone(),
            'net_fusion.fc.weight': model.state_dict()['net_fusion.fc.weight'].detach().clone(),
        }

        model.init_weights()
        after = model.state_dict()

        for key, tensor in before.items():
            assert torch.allclose(after[key], tensor)

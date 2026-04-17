"""Tests for AdaCLIPDetector."""

import importlib.util
import math
import os

import numpy as np
import pytest
import torch
from unittest import TestCase

from baoiad.datasets.transforms.augmentation import NormalizeAD
from baoiad.structures import ADDataSample

import baoiad  # noqa: F401
from baoiad.registry import MODELS


def _make_data_samples(batch_size, h=64, w=64):
    samples = []
    for i in range(batch_size):
        sample = ADDataSample()
        sample.gt_label = i % 2
        sample.gt_mask = torch.zeros(h, w)
        sample.cls_name = 'bottle'
        sample.img_path = f'/fake/{i}.png'
        sample.defect_type = 'good'
        samples.append(sample)
    return samples


def _has_clip_backend():
    return importlib.util.find_spec('open_clip') is not None


@pytest.mark.skipif(not _has_clip_backend(), reason='open_clip or local UniVAD clip backend unavailable')
class TestAdaCLIPDetector(TestCase):
    def setUp(self):
        self.cfg = dict(
            type='AdaCLIPDetector',
            clip_model='ViT-B-16',
            pretrained=None,
            image_size=64,
            features_list=[3, 6, 9, 12],
            prompting_depth=2,
            prompting_length=2,
            prompting_branch='VL',
            prompting_type='S',
            use_hsf=False,
            k_clusters=4,
            gaussian_sigma=0.0,
            official_checkpoint=None,
            enable_train_loss=True,
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
        assert 'loss' in out
        assert 'loss_cls' in out
        assert 'loss_seg' in out
        assert torch.isfinite(out['loss'])
        assert out['loss'].item() > 0

    def test_forward_predict(self):
        model = MODELS.build(self.cfg)
        model.eval()
        data_samples = _make_data_samples(2, 64, 64)
        out = model(torch.randn(2, 3, 64, 64), data_samples, mode='predict')
        assert isinstance(out, list)
        assert len(out) == 2
        for sample in out:
            assert math.isfinite(float(sample.pred_score))
            assert tuple(sample.pred_anomaly_map.shape[-2:]) == (64, 64)
            assert torch.isfinite(sample.pred_anomaly_map).all()

    def test_class_name_normalization(self):
        model = MODELS.build(self.cfg)
        samples = _make_data_samples(2, 64, 64)
        samples[0].cls_name = 'metal_nut'
        samples[1].cls_name = 'tooth-brush'
        classes = model._resolve_batch_classes(samples, batch_size=2)
        assert classes == ['metal nut', 'tooth brush']

    def test_missing_official_checkpoint_is_nonfatal_by_default(self):
        cfg = dict(self.cfg)
        cfg['official_checkpoint'] = os.path.join('/tmp', 'missing_adaclip_weights.pth')
        model = MODELS.build(cfg)
        model.eval()
        data_samples = _make_data_samples(1, 64, 64)
        out = model(torch.randn(1, 3, 64, 64), data_samples, mode='predict')
        assert len(out) == 1

    def test_missing_official_checkpoint_can_be_required(self):
        cfg = dict(self.cfg)
        cfg['official_checkpoint'] = os.path.join('/tmp', 'missing_adaclip_weights_required.pth')
        cfg['require_official_checkpoint'] = True
        with pytest.raises(FileNotFoundError):
            MODELS.build(cfg)

    def test_normalize_for_clip_matches_pixel_scale_conversion(self):
        model = MODELS.build(self.cfg)
        raw = np.array([[[255.0, 128.0, 0.0]]], dtype=np.float32)
        normalized = NormalizeAD().transform({'img': raw.copy()})['img']
        x = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0)
        out = model._normalize_for_clip(x)

        clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1) * 255.0
        clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1) * 255.0
        expected = (torch.from_numpy(raw.transpose(2, 0, 1)).unsqueeze(0) - clip_mean) / clip_std

        assert torch.allclose(out, expected, atol=1e-5)

    def test_only_official_trainables_are_unfrozen(self):
        model = MODELS.build(self.cfg)
        trainable = model.get_official_trainable_parameter_names()

        assert trainable
        assert 'clip.visual.positional_embedding' not in trainable
        assert model.clip.visual.positional_embedding.requires_grad is False
        assert all(
            any(
                fragment in name
                for fragment in (
                    'text_prompter',
                    'visual_prompter',
                    'patch_token_layer',
                    'cls_token_layer',
                    'dynamic_visual_prompt_generator',
                    'dynamic_text_prompt_generator',
                )
            )
            for name in trainable
        )

    def test_hsf_raises_on_non_finite_input(self):
        from baoiad.models.detectors.adaclip import HybridSemanticFusion

        hsf = HybridSemanticFusion(k_clusters=2)
        patch_tokens = [torch.randn(1, 4, 8), torch.randn(1, 4, 8)]
        anomaly_maps = [torch.randn(1, 4, 2), torch.randn(1, 4, 2)]
        patch_tokens[0][0, 0, 0] = float('nan')

        with pytest.raises(ValueError, match='Non-finite values detected'):
            hsf.forward(patch_tokens, anomaly_maps)

    def test_hsf_tolerates_reduced_cluster_count(self):
        from baoiad.models.detectors.adaclip import HybridSemanticFusion

        hsf = HybridSemanticFusion(k_clusters=4)
        repeated = torch.ones(1, 4, 8)
        patch_tokens = [repeated.clone(), repeated.clone()]
        anomaly_maps = [torch.zeros(1, 4, 2), torch.zeros(1, 4, 2)]

        out = hsf.forward(patch_tokens, anomaly_maps)
        assert out.shape == (1, 8)
        assert torch.isfinite(out).all()

    def test_extract_image_features_no_prompts_is_finite(self):
        model = MODELS.build(self.cfg)
        model.eval()
        x = model._normalize_for_clip(torch.randn(1, 3, 64, 64))
        pooled = model.adaclip._extract_image_features_no_prompts(x.float())
        assert torch.isfinite(pooled).all()

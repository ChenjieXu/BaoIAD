"""Tests for NSADetector."""

from unittest import TestCase

import numpy as np
import pytest
import torch
import torch.nn as nn

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


def _make_cls_samples(batch_size, cls_name, H=256, W=256):
    samples = _make_data_samples(batch_size, H=H, W=W)
    for sample in samples:
        sample.cls_name = cls_name
    return samples

class TestNSADetector(TestCase):
    def setUp(self):
        self.cfg = dict(type='NSADetector', backbone='resnet18', anomaly_ratio=1.0, seg_base_width=32)

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

    def test_forward_loss_prefers_dataset_masks(self):
        model = MODELS.build(self.cfg)
        model.train()
        data_samples = _make_data_samples(2, 64, 64)
        model._generate_nsa = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError('_generate_nsa should not be called when gt_mask is provided'))
        out = model(torch.randn(2, 3, 64, 64), data_samples, mode='loss')
        assert 'loss' in out

    def test_forward_predict(self):
        model = MODELS.build(self.cfg)
        model.eval()
        data_samples = _make_data_samples(2, 64, 64)
        out = model(torch.randn(2, 3, 64, 64), data_samples, mode='predict')
        assert isinstance(out, list)
        assert len(out) == 2

    def test_loss_fn_is_bce_with_logits(self):
        model = MODELS.build(self.cfg)
        assert isinstance(model.loss_fn, nn.BCEWithLogitsLoss)
        assert len(model.model.uplayer1) == 1
        assert len(model.model.uplayer2) == 1
        assert len(model.model.uplayer3) == 1

    def test_predict_scores_in_unit_interval(self):
        """Predict mode uses sigmoid, so all scores must lie in [0, 1]."""
        model = MODELS.build(self.cfg)
        model.eval()
        data_samples = _make_data_samples(2, 64, 64)
        out = model(torch.randn(2, 3, 64, 64), data_samples, mode='predict')
        for sample in out:
            assert 0 <= sample.pred_score <= 1
            assert ((sample.pred_anomaly_map >= 0) & (sample.pred_anomaly_map <= 1)).all()

    def test_object_full_size_predict_keeps_256_maps(self):
        model = MODELS.build(self.cfg)
        model.eval()
        data_samples = _make_cls_samples(2, cls_name='bottle', H=256, W=256)
        out = model(torch.randn(2, 3, 256, 256), data_samples, mode='predict')
        for sample in out:
            assert sample.pred_anomaly_map.shape == (1, 256, 256)
            assert 0 <= sample.pred_score <= 1

    def test_texture_predict_uses_mean_score(self):
        model = MODELS.build(self.cfg)
        model.eval()
        data_samples = _make_cls_samples(2, cls_name='tile', H=256, W=256)
        out = model(torch.randn(2, 3, 256, 256), data_samples, mode='predict')
        for sample in out:
            assert sample.pred_anomaly_map.shape == (1, 256, 256)
            assert sample.pred_score == pytest.approx(
                sample.pred_anomaly_map.mean().item(), rel=1e-5, abs=1e-6)

    def test_object_predict_score_uses_unpadded_crop_mean(self):
        model = MODELS.build(self.cfg)
        model.eval()
        inputs = torch.randn(2, 3, 256, 256)
        data_samples = _make_cls_samples(2, cls_name='bottle', H=256, W=256)
        out = model(inputs, data_samples, mode='predict')
        score_map, img_scores = model._predict_score_map(inputs, 'bottle')
        for i, sample in enumerate(out):
            assert sample.pred_score == pytest.approx(float(img_scores[i].item()), rel=1e-5, abs=1e-6)
            assert sample.pred_score != pytest.approx(
                score_map[i].mean().item(), rel=1e-5, abs=1e-6)

    def test_compute_image_scores_from_map_reference_mean_matches_object_crop(self):
        score_map = torch.zeros(1, 1, 256, 256)
        score_map[:, :, 16:240, 16:240] = 1.0
        score_map[:, :, :16, :] = 5.0
        score_map[:, :, 240:, :] = 5.0
        score_map[:, :, :, :16] = 5.0
        score_map[:, :, :, 240:] = 5.0

        score = MODELS.build(self.cfg)._compute_image_scores_from_map(
            score_map,
            cls_name='bottle',
            input_hw=(256, 256),
            mode='reference_mean',
        )
        assert score.item() == pytest.approx(1.0, rel=1e-6, abs=1e-6)

    def test_compute_image_scores_from_map_full_mean_uses_full_map(self):
        score_map = torch.zeros(1, 1, 256, 256)
        score_map[:, :, :16, :] = 1.0

        score = MODELS.build(self.cfg)._compute_image_scores_from_map(
            score_map,
            cls_name='bottle',
            input_hw=(256, 256),
            mode='full_mean',
        )
        expected = float(score_map.mean().item())
        assert score.item() == pytest.approx(expected, rel=1e-6, abs=1e-6)

    def test_compute_image_scores_from_map_topk_mean_uses_reference_region(self):
        score_map = torch.zeros(1, 1, 256, 256)
        score_map[:, :, 16:240, 16:240] = 0.2
        score_map[:, :, 100:104, 100:104] = 0.9
        score_map[:, :, :16, :] = 3.0

        score = MODELS.build(self.cfg)._compute_image_scores_from_map(
            score_map,
            cls_name='bottle',
            input_hw=(256, 256),
            mode='topk_mean',
            topk_ratio=16 / float(224 * 224),
        )
        assert score.item() == pytest.approx(0.9, rel=1e-4, abs=1e-4)

    def test_compute_image_scores_from_map_texture_reference_mean_uses_full_map(self):
        score_map = torch.zeros(1, 1, 256, 256)
        score_map[:, :, :16, :] = 1.0

        score = MODELS.build(self.cfg)._compute_image_scores_from_map(
            score_map,
            cls_name='tile',
            input_hw=(256, 256),
            mode='reference_mean',
        )
        expected = float(score_map.mean().item())
        assert score.item() == pytest.approx(expected, rel=1e-6, abs=1e-6)


def test_generate_nsa_uses_category_specific_clone_mode(monkeypatch):
    from baoiad.models.detectors import nsa as nsa_module

    record = []

    def fake_patch_ex(**kwargs):
        record.append(kwargs['mode'])
        dest = kwargs['ima_dest']
        label = np.zeros(dest.shape[:2] + (1,), dtype=np.float32)
        return dest.copy(), label

    monkeypatch.setattr(nsa_module, '_official_patch_ex', fake_patch_ex)

    model = MODELS.build(dict(type='NSADetector', backbone='resnet18', anomaly_ratio=1.0))
    model.train()

    for cls_name in ['bottle', 'tile']:
        record.clear()
        model._current_cls_name = cls_name
        _ = model._generate_nsa(torch.rand(2, 3, 64, 64))
        expected = nsa_module.CV2_NORMAL_CLONE if cls_name == 'bottle' else nsa_module.CV2_MIXED_CLONE
        assert len(record) == 2
        assert all(mode == expected for mode in record)

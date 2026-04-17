"""Tests for ASTDetector."""

from unittest import TestCase
from unittest.mock import patch

import pytest
import torch

from baoiad.structures import ADDataSample

import baoiad  # noqa: F401
from baoiad.registry import MODELS


def _make_data_samples(batch_size, H=64, W=64):
    samples = []
    for i in range(batch_size):
        s = ADDataSample()
        s.gt_label = i % 2
        s.gt_mask = torch.zeros(H, W)
        s.cls_name = 'bottle'
        s.img_path = f'/fake/{i}.png'
        s.defect_type = 'good' if i % 2 == 0 else 'broken'
        samples.append(s)
    return samples


try:
    import timm  # noqa: F401
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


@pytest.mark.skipif(not HAS_TIMM, reason='timm not installed')
class TestASTDetector(TestCase):
    def setUp(self):
        self.cfg = dict(
            type='ASTDetector',
            backbone=dict(model_name='tf_efficientnet_b5', pretrained=False),
            extract_layer=35,
            n_feat=8,
            map_len=4,
            n_coupling_blocks=2,
            channels_hidden_teacher=16,
            channels_hidden_student=32,
            n_student_blocks=2,
            img_size=64,
            score_map_size=16,
            pos_enc=True,
            pos_enc_dim=8,
        )

    def _mock_feats(self, model, batch_size):
        return torch.randn(batch_size, model.n_feat, model.map_len, model.map_len)

    def test_forward_tensor(self):
        model = MODELS.build(self.cfg)
        model.eval()
        with patch.object(model, 'extract_features', return_value=self._mock_feats(model, 2)):
            out = model(torch.randn(2, 3, 64, 64), mode='tensor')
        assert out is not None

    def test_forward_loss_joint_returns_teacher_and_student_losses(self):
        model = MODELS.build(self.cfg)
        model.train()
        data_samples = _make_data_samples(2, 64, 64)
        with patch.object(model, 'extract_features', return_value=self._mock_feats(model, 2)):
            out = model(torch.randn(2, 3, 64, 64), data_samples, mode='loss')
        assert set(out.keys()) == {'loss', 'loss_teacher', 'loss_student'}
        assert torch.isfinite(out['loss'])

    def test_forward_predict_student_exposes_mean_and_max_scores(self):
        model = MODELS.build(dict(**self.cfg, training_phase='student'))
        model.eval()
        data_samples = _make_data_samples(2, 64, 64)
        with patch.object(model, 'extract_features', return_value=self._mock_feats(model, 2)):
            out = model(torch.randn(2, 3, 64, 64), data_samples, mode='predict')
        assert isinstance(out, list)
        assert len(out) == 2
        assert hasattr(out[0], 'pred_score')
        assert hasattr(out[0], 'pred_score_mean')
        assert hasattr(out[0], 'pred_score_max')
        assert out[0].pred_score == pytest.approx(out[0].pred_score_max)
        assert out[0].pred_anomaly_map.shape == (1, 16, 16)

    def test_forward_predict_mean_mode_uses_mean_score_as_primary(self):
        model = MODELS.build(dict(**self.cfg, training_phase='student', image_score_mode='mean'))
        model.eval()
        data_samples = _make_data_samples(2, 64, 64)
        with patch.object(model, 'extract_features', return_value=self._mock_feats(model, 2)):
            out = model(torch.randn(2, 3, 64, 64), data_samples, mode='predict')
        assert out[0].pred_score == pytest.approx(out[0].pred_score_mean)

    def test_teacher_phase_only_returns_teacher_loss(self):
        model = MODELS.build(dict(**self.cfg, training_phase='teacher'))
        model.train()
        data_samples = _make_data_samples(2, 64, 64)
        with patch.object(model, 'extract_features', return_value=self._mock_feats(model, 2)):
            out = model(torch.randn(2, 3, 64, 64), data_samples, mode='loss')
        assert set(out.keys()) == {'loss', 'loss_teacher'}
        assert all(not param.requires_grad for param in model.student.parameters())

    def test_teacher_phase_predict_uses_teacher_scores(self):
        model = MODELS.build(dict(**self.cfg, training_phase='teacher'))
        model.eval()
        data_samples = _make_data_samples(2, 64, 64)
        with patch.object(model, 'extract_features', return_value=self._mock_feats(model, 2)):
            out = model(torch.randn(2, 3, 64, 64), data_samples, mode='predict')
        assert hasattr(out[0], 'pred_score_mean')
        assert hasattr(out[0], 'pred_score_max')

"""Tests for SuperSimpleNetDetector."""

from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import torch
from mmengine import Config

from baoiad.structures import ADDataSample

import baoiad  # noqa: F401
from baoiad.models.detectors.supersimplenet import AnomalyGenerator
from baoiad.registry import MODELS

ROOT = Path(__file__).resolve().parents[3]


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


class TestSuperSimpleNetDetector(TestCase):
    def setUp(self):
        self.cfg = dict(
            type='SuperSimpleNetDetector',
            backbone='resnet18',
            layers=['layer2', 'layer3'],
            sigma=4.0,
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

    def test_predict_outputs_are_sigmoid_bounded(self):
        model = MODELS.build(self.cfg)
        model.eval()
        data_samples = _make_data_samples(2, 64, 64)
        out = model(torch.randn(2, 3, 64, 64), data_samples, mode='predict')

        for sample in out:
            assert 0.0 <= sample.pred_score <= 1.0
            assert 0.0 <= sample.pred_score_mean <= 1.0
            assert 0.0 <= sample.pred_score_max <= 1.0
            assert torch.isfinite(sample.pred_anomaly_map).all()
            assert (sample.pred_anomaly_map >= 0).all()
            assert (sample.pred_anomaly_map <= 1).all()
            assert sample.pred_score_max >= sample.pred_score_mean

    def test_loss_path_uses_raw_features_for_cls_by_default(self):
        model = MODELS.build(self.cfg)
        model.train()
        data_samples = _make_data_samples(2, 64, 64)
        captured = {}

        def fake_forward(input_features, adapted_features, mask, labels):
            captured['input_features_is_none'] = input_features is None
            return input_features, adapted_features, mask, labels

        with patch.object(model.anomaly_generator, 'forward', side_effect=fake_forward):
            losses = model(torch.randn(2, 3, 64, 64), data_samples, mode='loss')

        assert captured['input_features_is_none'] is False
        assert 'loss' in losses
        assert torch.isfinite(losses['loss'])

    def test_loss_path_drops_raw_features_when_adapt_cls_features_enabled(self):
        model = MODELS.build(dict(**self.cfg, adapt_cls_features=True))
        model.train()
        data_samples = _make_data_samples(2, 64, 64)
        captured = {}

        def fake_forward(input_features, adapted_features, mask, labels):
            captured['input_features_is_none'] = input_features is None
            return None, adapted_features, mask, labels

        with patch.object(model.anomaly_generator, 'forward', side_effect=fake_forward):
            losses = model(torch.randn(2, 3, 64, 64), data_samples, mode='loss')

        assert captured['input_features_is_none'] is True
        assert 'loss' in losses
        assert torch.isfinite(losses['loss'])

    def test_strict_config_freezes_official_optimizer_groups(self):
        cfg = Config.fromfile(str(ROOT / 'configs' / 'supersimplenet' / 'supersimplenet_256_mvtec_strict.py'))

        assert cfg.benchmark_result_selector == {'mode': 'last'}
        assert cfg.model['adapt_cls_features'] is False
        assert tuple(cfg.model['layers']) == ('layer2', 'layer3')
        assert cfg.optim_wrapper.optimizer['type'] == 'AdamW'
        assert cfg.optim_wrapper.optimizer['lr'] == 2e-4
        assert cfg.optim_wrapper.optimizer['weight_decay'] == 1e-5
        assert cfg.optim_wrapper.paramwise_cfg['custom_keys']['adaptor']['lr_mult'] == 0.5
        assert cfg.optim_wrapper.paramwise_cfg['custom_keys']['adaptor']['decay_mult'] == 1000.0
        assert cfg.param_scheduler[0]['type'] == 'MultiStepLR'
        assert cfg.param_scheduler[0]['milestones'] == [240, 270]
        assert cfg.param_scheduler[0]['gamma'] == 0.4
        assert cfg.train_cfg.max_epochs == 300
        assert cfg.test_evaluator['image_score_field'] == 'pred_score_max'

    def test_official_perlin_mask_is_binary(self):
        generator = AnomalyGenerator(threshold=0.2)
        torch.manual_seed(42)
        mask = generator._generate_perlin_mask(2, 16, 16, device=torch.device('cpu'))

        assert mask.shape == (2, 1, 16, 16)
        assert set(mask.unique().tolist()).issubset({0.0, 1.0})

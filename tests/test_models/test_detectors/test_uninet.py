"""Tests for UniNetDetector."""

from pathlib import Path
from unittest import TestCase

import torch
from mmengine import Config

import baoiad  # noqa: F401
from baoiad.registry import MODELS
from baoiad.structures import ADDataSample

ROOT = Path(__file__).resolve().parents[3]


def _make_data_samples(batch_size, height=64, width=64):
    samples = []
    for i in range(batch_size):
        sample = ADDataSample()
        sample.gt_label = i % 2
        sample.gt_mask = torch.zeros(height, width)
        sample.cls_name = 'bottle'
        sample.img_path = f'/fake/{i}.png'
        sample.defect_type = 'good'
        samples.append(sample)
    return samples


class TestUniNetDetector(TestCase):
    def setUp(self):
        self.cfg = dict(
            type='UniNetDetector',
            teacher_backbone=dict(
                type='FeatureExtractor',
                backbone_name='wide_resnet50_2',
                pretrained=False,
                out_indices=(1, 2, 3),
                frozen=False,
            ),
        )

    def test_forward_tensor(self):
        model = MODELS.build(self.cfg)
        model.eval()

        out = model(torch.randn(2, 3, 64, 64), mode='tensor')

        assert isinstance(out, list)
        assert len(out) == 6
        assert [tuple(feature.shape) for feature in out] == [
            (2, 256, 16, 16),
            (2, 512, 8, 8),
            (2, 1024, 4, 4),
            (2, 256, 16, 16),
            (2, 512, 8, 8),
            (2, 1024, 4, 4),
        ]
        assert all(torch.isfinite(feature).all() for feature in out)

    def test_forward_loss(self):
        model = MODELS.build(self.cfg)
        model.train()
        data_samples = _make_data_samples(2, 64, 64)

        out = model(torch.randn(2, 3, 64, 64), data_samples, mode='loss')

        assert sorted(out.keys()) == ['loss']
        assert torch.isfinite(out['loss']).all()

    def test_forward_predict(self):
        model = MODELS.build(self.cfg)
        model.eval()
        data_samples = _make_data_samples(2, 64, 64)

        out = model(torch.randn(2, 3, 64, 64), data_samples, mode='predict')

        assert isinstance(out, list)
        assert len(out) == 2
        assert all(hasattr(result, 'pred_score') for result in out)
        maps = torch.stack([result.pred_anomaly_map for result in out])
        scores = torch.tensor([float(torch.as_tensor(result.pred_score).detach().cpu()) for result in out])
        assert tuple(maps.shape) == (2, 1, 64, 64)
        assert torch.isfinite(maps).all()
        assert torch.isfinite(scores).all()

    def test_teacher_backbone_cfg_preserves_feature_extractor_settings(self):
        model = MODELS.build(self.cfg)
        model.train()

        assert model.teacher_backbone_name == 'wide_resnet50_2'
        assert tuple(model.teacher_backbone_cfg['out_indices']) == (1, 2, 3)
        assert model.teacher_backbone_cfg['pretrained'] is False
        assert tuple(model.teachers.target_teacher.out_indices) == (1, 2, 3)
        assert model.teachers.source_teacher.training is False
        assert all(not parameter.requires_grad for parameter in model.teachers.source_teacher.parameters())
        assert any(parameter.requires_grad for parameter in model.teachers.target_teacher.parameters())


def test_uninet_strict_config_matches_anomalib_runtime():
    cfg = Config.fromfile(ROOT / 'configs' / 'uninet' / 'uninet_256_mvtec_strict.py')

    assert cfg.benchmark_multi_class is False
    assert cfg.benchmark_keep_dataloader_workers is True
    assert cfg.benchmark_preserve_checkpoint_hooks is True
    assert cfg.benchmark_result_selector == {'mode': 'best', 'metric': 'image_auroc'}
    assert cfg.train_dataloader.batch_size == 32
    assert cfg.test_dataloader.batch_size == 32
    assert cfg.train_dataloader.num_workers == 8
    assert cfg.test_dataloader.num_workers == 8
    assert cfg.model.type == 'UniNetDetector'
    assert cfg.model.teacher_backbone.type == 'FeatureExtractor'
    assert cfg.model.teacher_backbone.backbone_name == 'wide_resnet50_2'
    assert tuple(cfg.model.teacher_backbone.out_indices) == (1, 2, 3)
    assert cfg.model.lambda_weight == 0.7
    assert cfg.model.temperature == 0.1
    assert cfg.optim_wrapper.optimizer.type == 'AdamW'
    assert cfg.optim_wrapper.optimizer.lr == 5e-3
    assert cfg.optim_wrapper.optimizer.weight_decay == 1e-5
    assert cfg.optim_wrapper.optimizer.eps == 1e-10
    assert cfg.optim_wrapper.optimizer.amsgrad is True
    assert cfg.optim_wrapper.paramwise_cfg.custom_keys['teachers.target_teacher']['lr_mult'] == 2e-4
    assert cfg.optim_wrapper.paramwise_cfg.custom_keys['fc']['lr_mult'] == 0.0
    assert cfg.optim_wrapper.paramwise_cfg.custom_keys['fc']['decay_mult'] == 0.0
    assert cfg.param_scheduler[0].type == 'MultiStepLR'
    assert list(cfg.param_scheduler[0].milestones) == [80]
    assert cfg.param_scheduler[0].gamma == 0.2
    assert cfg.train_cfg.max_epochs == 100
    assert cfg.train_cfg.val_interval == 1
    assert any(hook['type'] == 'EarlyStoppingHook' for hook in cfg.custom_hooks)

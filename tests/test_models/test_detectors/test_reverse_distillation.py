"""Tests for ReverseDistillation."""

from pathlib import Path
from unittest import TestCase

import torch
from mmengine import Config

import baoiad  # noqa: F401
from baoiad.models.detectors.reverse_distillation import ReverseDistillation
from baoiad.registry import MODELS
from baoiad.structures import ADDataSample

ROOT = Path(__file__).resolve().parents[3]


def _make_data_samples(batch_size, H=256, W=256):
    samples = []
    for i in range(batch_size):
        sample = ADDataSample()
        sample.gt_label = i % 2
        sample.gt_mask = torch.zeros(H, W)
        sample.cls_name = 'bottle'
        sample.img_path = f'/fake/{i}.png'
        sample.defect_type = 'good'
        samples.append(sample)
    return samples


class TestReverseDistillation(TestCase):
    def setUp(self):
        self.cfg = dict(
            type='ReverseDistillation',
            backbone=dict(
                type='FeatureExtractor',
                backbone_name='wide_resnet50_2',
                pretrained=False,
                out_indices=(1, 2, 3),
                frozen=True,
            ),
            anomaly_map_mode='add',
            smooth_sigma=4.0,
            smoothing_backend='scipy',
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
        assert torch.isfinite(out['loss']).all()

    def test_forward_predict(self):
        model = MODELS.build(self.cfg)
        model.eval()
        data_samples = _make_data_samples(2, 64, 64)
        out = model(torch.randn(2, 3, 64, 64), data_samples, mode='predict')
        assert isinstance(out, list)
        assert len(out) == 2

        maps = torch.stack([result.pred_anomaly_map for result in out])
        scores = torch.tensor([float(result.pred_score) for result in out])
        assert tuple(maps.shape) == (2, 1, 64, 64)
        assert torch.isfinite(maps).all()
        assert torch.isfinite(scores).all()


def test_rd_flatten_cosine_loss_matches_official_formula():
    teacher_features = [
        torch.tensor([[[[1.0, 0.0]]]]),
        torch.tensor([[[[1.0, 2.0]]]]),
    ]
    student_features = [
        torch.tensor([[[[0.0, 1.0]]]]),
        torch.tensor([[[[1.0, 1.0]]]]),
    ]

    loss = ReverseDistillation._flatten_cosine_loss(teacher_features, student_features)

    cosine = torch.nn.CosineSimilarity()
    expected = (
        torch.mean(1 - cosine(teacher_features[0].view(1, -1), student_features[0].view(1, -1)))
        + torch.mean(1 - cosine(teacher_features[1].view(1, -1), student_features[1].view(1, -1)))
    )
    assert torch.allclose(loss, expected)


def test_rd_predict_uses_additive_anomaly_map_and_optional_smoothing():
    model = MODELS.build(dict(
        type='ReverseDistillation',
        backbone=dict(
            type='FeatureExtractor',
            backbone_name='wide_resnet50_2',
            pretrained=False,
            out_indices=(1, 2, 3),
            frozen=True,
        ),
        anomaly_map_mode='add',
        smooth_sigma=0.0,
        smoothing_backend='scipy',
    ))
    model.eval()

    teacher_features = [
        torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]),
        torch.tensor([[[[1.0]]]]),
    ]
    student_features = [
        torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]]),
        torch.tensor([[[[-1.0]]]]),
    ]

    anomaly_map = model._compute_anomaly_map(teacher_features, student_features, image_size=(2, 2))
    expected = torch.tensor([[[3.0, 3.0], [3.0, 3.0]]])
    assert torch.allclose(anomaly_map, expected)

    smoothed = model._smooth_anomaly_map(anomaly_map)
    assert torch.allclose(smoothed, expected)


def test_rd_strict_config_matches_official_reference_settings():
    cfg = Config.fromfile(ROOT / 'configs' / 'rd' / 'rd_wrn50_256_mvtec_strict.py')

    assert cfg.randomness.seed == 111
    assert cfg.randomness.deterministic is True
    assert cfg.benchmark_multi_class is False
    assert cfg.benchmark_keep_dataloader_workers is True
    assert cfg.benchmark_result_selector.mode == 'last'
    assert cfg.train_dataloader.batch_size == 16
    assert cfg.train_dataloader.num_workers == 4
    assert cfg.train_dataloader.persistent_workers is True
    assert cfg.test_dataloader.batch_size == 1
    assert cfg.test_dataloader.num_workers == 4
    assert cfg.test_dataloader.persistent_workers is True
    assert cfg.model.type == 'ReverseDistillation'
    assert cfg.model.backbone.type == 'FeatureExtractor'
    assert cfg.model.backbone.backbone_name == 'wide_resnet50_2'
    assert tuple(cfg.model.backbone.out_indices) == (1, 2, 3)
    assert cfg.model.anomaly_map_mode == 'add'
    assert cfg.model.smooth_sigma == 4.0
    assert cfg.model.smoothing_backend == 'scipy'
    assert cfg.optim_wrapper.optimizer.type == 'Adam'
    assert cfg.optim_wrapper.optimizer.lr == 0.005
    assert tuple(cfg.optim_wrapper.optimizer.betas) == (0.5, 0.999)
    assert cfg.optim_wrapper.optimizer.weight_decay == 0.0
    assert cfg.param_scheduler == []
    assert cfg.train_cfg.max_epochs == 200
    assert cfg.train_cfg.val_begin == 10
    assert cfg.train_cfg.val_interval == 10

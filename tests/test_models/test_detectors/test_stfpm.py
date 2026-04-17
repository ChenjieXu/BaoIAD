"""Tests for STFPMDetector."""

from pathlib import Path
from unittest import TestCase

import cv2
import pytest
import torch
import torch.nn.functional as F
from mmengine import Config
from baoiad.structures import ADDataSample

import baoiad  # noqa: F401
from baoiad.models.detectors.stfpm import STFPMDetector
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


class TestSTFPMDetector(TestCase):
    def setUp(self):
        self.cfg = dict(type='STFPMDetector', backbone='resnet18')

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


def test_stfpm_official_loss_matches_reference_formula():
    model = STFPMDetector.__new__(STFPMDetector)
    model.reference_impl = 'official'

    teacher_features = [
        torch.tensor([[[[1.0, 0.0]]]]),
        torch.tensor([[[[1.0, 2.0]]]]),
    ]
    student_features = [
        torch.tensor([[[[0.0, 1.0]]]]),
        torch.tensor([[[[1.0, 1.0]]]]),
    ]

    loss = STFPMDetector._compute_loss(model, teacher_features, student_features)
    expected = sum(
        torch.sum((F.normalize(t_feat, p=2, dim=1) - F.normalize(s_feat, p=2, dim=1)) ** 2, dim=1).mean()
        for t_feat, s_feat in zip(teacher_features, student_features)
    )
    assert torch.allclose(loss, expected)


def test_stfpm_official_anomaly_map_matches_reference_formula():
    model = STFPMDetector.__new__(STFPMDetector)
    model.reference_impl = 'official'

    teacher_features = [
        torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]),
        torch.tensor([[[[1.0]]]]),
    ]
    student_features = [
        torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]]),
        torch.tensor([[[[-1.0]]]]),
    ]

    anomaly_map = STFPMDetector._compute_anomaly(
        model,
        teacher_features,
        student_features,
        target_size=(2, 2),
    )
    expected = torch.full((1, 1, 2, 2), 4.0)
    assert torch.allclose(anomaly_map, expected)


def test_stfpm_official_predict_resizes_map_for_image_scoring():
    model = STFPMDetector.__new__(STFPMDetector)
    model.image_score_mode = 'map_max'
    model.image_score_mode_overrides = {}
    raw_score_map = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]])
    data_samples = _make_data_samples(1, 4, 4)
    inputs = torch.zeros(1, 3, 4, 4)

    outputs = STFPMDetector._build_official_predict_outputs(
        model,
        raw_score_map,
        inputs,
        data_samples,
    )

    assert len(outputs) == 1
    result = outputs[0]
    assert tuple(result.pred_anomaly_map.shape) == (1, 4, 4)
    assert tuple(result.pred_anomaly_map_raw.shape) == (1, 2, 2)
    expected_map = torch.from_numpy(
        cv2.resize(raw_score_map[0, 0].numpy().astype('float32'), (4, 4), interpolation=cv2.INTER_LINEAR)
    ).unsqueeze(0)
    assert torch.allclose(result.pred_anomaly_map, expected_map)
    assert torch.allclose(result.pred_anomaly_map_raw, raw_score_map[0])
    assert float(result.pred_score) == pytest.approx(float(expected_map.max()))


def test_stfpm_strict_config_matches_official_protocol():
    cfg = Config.fromfile(ROOT / 'configs' / 'stfpm' / 'stfpm_rn18_256_mvtec_strict.py')

    assert cfg.benchmark_multi_class is False
    assert cfg.benchmark_preserve_checkpoint_hooks is True
    assert cfg.benchmark_resume_existing is True
    assert cfg.benchmark_test_after_train is True
    assert cfg.benchmark_checkpoint_source == 'best'
    assert cfg.benchmark_result_selector.mode == 'last'
    assert cfg.randomness.seed == 0
    assert cfg.train_dataloader.batch_size == 32
    assert cfg.val_dataloader.batch_size == 32
    assert cfg.test_dataloader.batch_size == 1
    assert cfg.train_dataloader.dataset['split'] == 'train'
    assert cfg.train_dataloader.dataset['train_val_split_ratio'] == 0.2
    assert cfg.train_dataloader.dataset['train_val_split_subset'] == 'train'
    assert cfg.val_dataloader.dataset['split'] == 'train'
    assert cfg.val_dataloader.dataset['train_val_split_subset'] == 'val'
    assert cfg.test_dataloader.dataset['split'] == 'test'
    assert cfg.val_evaluator.type == 'AnomalyMapMeanMetric'
    assert cfg.benchmark_timeout == 14400
    assert cfg.default_hooks.checkpoint.save_best == 'ad/score_mean'
    assert cfg.default_hooks.checkpoint.rule == 'less'
    assert cfg.default_hooks.checkpoint.save_last is False
    assert cfg.default_hooks.checkpoint.interval == 1000000
    assert cfg.model.type == 'STFPMDetector'
    assert cfg.model.reference_impl == 'official'
    assert cfg.model.backbone.type == 'FeatureExtractor'
    assert cfg.model.backbone.backbone_name == 'resnet18'
    assert tuple(cfg.model.backbone.out_indices) == (1, 2, 3)
    assert cfg.optim_wrapper.optimizer.type == 'SGD'
    assert cfg.optim_wrapper.optimizer.lr == 0.4
    assert cfg.optim_wrapper.optimizer.weight_decay == 1e-4
    assert cfg.param_scheduler == []
    assert cfg.train_cfg.max_epochs == 100
    assert cfg.train_cfg.val_begin == 1
    assert cfg.train_cfg.val_interval == 1


def test_stfpm_image_score_modes_match_manual_formula():
    model = STFPMDetector.__new__(STFPMDetector)
    model.image_score_mode = 'map_p99'
    model.image_score_mode_overrides = {'zipper': 'map_mean'}

    score_map = torch.tensor([
        [[[0.0, 1.0], [2.0, 3.0]]],
        [[[1.0, 1.0], [5.0, 5.0]]],
    ], dtype=torch.float32)
    raw_score_map = score_map.clone()
    data_samples = _make_data_samples(2, 2, 2)
    data_samples[1].cls_name = 'zipper'

    outputs = STFPMDetector._compute_image_scores(
        model,
        score_map,
        raw_score_map=raw_score_map,
        data_samples=data_samples,
    )

    expected_p99 = torch.quantile(score_map[0].view(-1), 0.99).item()
    expected_mean = score_map[1].mean().item()
    assert float(outputs['pred_score'][0]) == pytest.approx(expected_p99)
    assert float(outputs['pred_score'][1]) == pytest.approx(expected_mean)
    assert torch.allclose(outputs['pred_score_mean'], score_map.view(2, -1).mean(dim=1))
    assert torch.allclose(outputs['pred_score_max'], score_map.view(2, -1).max(dim=1).values)


def test_stfpm_invalid_image_score_mode_raises():
    with pytest.raises(ValueError, match='Unsupported STFPM image_score_mode'):
        MODELS.build(dict(type='STFPMDetector', backbone='resnet18', image_score_mode='bad_mode'))

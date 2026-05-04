"""Tests for AnomalyDINODetector."""

import math
from copy import deepcopy
from unittest import TestCase

import torch

import baoiad  # noqa: F401
from baoiad.registry import MODELS
from baoiad.structures import ADDataSample


def _make_data_samples(batch_size, cls_name='bottle', start_index=0, h=56, w=56):
    samples = []
    for index in range(batch_size):
        sample = ADDataSample()
        sample.gt_label = index % 2
        sample.gt_mask = torch.zeros(h, w)
        sample.cls_name = cls_name
        sample.img_path = f'/fake/{start_index + index:03d}.png'
        sample.defect_type = 'good'
        samples.append(sample)
    return samples


class _FakeFewShotDataset:
    def __init__(self, num_samples=4, cls_name='bottle'):
        self.requested_indices = []
        self.infos = [
            dict(
                img_path=f'/fake/{index:03d}.png',
                cls_name=cls_name,
                defect_type='good',
                gt_label=0,
            )
            for index in range(num_samples)
        ]

    def __len__(self):
        return len(self.infos)

    def full_init(self):
        return None

    def get_data_info(self, index):
        return dict(self.infos[index])

    def __getitem__(self, index):
        self.requested_indices.append(index)
        sample = ADDataSample()
        sample.cls_name = self.infos[index]['cls_name']
        sample.img_path = self.infos[index]['img_path']
        sample.defect_type = self.infos[index]['defect_type']
        sample.gt_label = self.infos[index]['gt_label']
        sample.gt_mask = torch.zeros(56, 56)
        return dict(
            inputs=torch.zeros(3, 56, 56),
            data_samples=sample,
        )


class _FakeLoader:
    def __init__(self, dataset):
        self.dataset = dataset


class TestAnomalyDINODetector(TestCase):
    """Test AnomalyDINODetector with pretrained=False for fast testing."""

    def setUp(self):
        self.cfg = dict(
            type='AnomalyDINODetector',
            backbone=dict(
                type='DINOv2Backbone',
                model_name='dinov2_vits14',
                frozen=True,
                pretrained=False,
            ),
            k=1,
            pca_foreground=False,
            top_ratio=0.01,
            gaussian_sigma=4.0,
        )

    def test_forward_tensor(self):
        model = MODELS.build(self.cfg)
        model.eval()
        out = model(torch.randn(2, 3, 56, 56), mode='tensor')
        assert out.shape[:2] == (2, 16)

    def test_forward_loss(self):
        model = MODELS.build(self.cfg)
        model.train()
        out = model(torch.randn(2, 3, 56, 56), _make_data_samples(2), mode='loss')
        assert isinstance(out, dict)
        assert 'loss' in out

    def test_forward_predict_without_memory_bank(self):
        model = MODELS.build(self.cfg)
        model.eval()
        with self.assertRaises(RuntimeError):
            model(torch.randn(2, 3, 56, 56), _make_data_samples(2), mode='predict')

    def test_forward_predict_with_memory_bank(self):
        model = MODELS.build(self.cfg)
        model.train()

        for step in range(5):
            data_samples = _make_data_samples(2, start_index=step * 2)
            model(torch.randn(2, 3, 56, 56), data_samples, mode='loss')

        model.build_memory_bank()
        assert model.memory_bank.shape[0] > 0

        model.eval()
        outputs = model(torch.randn(2, 3, 56, 56), _make_data_samples(2), mode='predict')
        assert len(outputs) == 2
        assert math.isfinite(outputs[0].pred_score)
        assert torch.isfinite(outputs[0].pred_anomaly_map).all()

    def test_agnostic_preprocess_uses_official_masking_categories(self):
        cfg = deepcopy(self.cfg)
        cfg.update(preprocess='agnostic')
        model = MODELS.build(cfg)

        assert model._should_apply_masking('capsule') is True
        assert model._should_apply_masking('bottle') is False
        assert model._rotation_angles_for_category('bottle') == (45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)

    def test_informed_preprocess_rotates_only_selected_categories(self):
        cfg = deepcopy(self.cfg)
        cfg.update(preprocess='informed')
        model = MODELS.build(cfg)

        assert model._rotation_angles_for_category('hazelnut') == (45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)
        assert model._rotation_angles_for_category('bottle') == ()

    def test_background_mask_has_expected_shape(self):
        cfg = deepcopy(self.cfg)
        cfg.update(preprocess='agnostic')
        model = MODELS.build(cfg)

        tokens = torch.randn(16, 384)
        mask = model._build_background_mask(tokens, spatial_h=4, spatial_w=4)

        assert mask.dtype == torch.bool
        assert tuple(mask.shape) == (16,)
        assert mask.any()

    def test_few_shot_selection_matches_official_sorted_slice(self):
        cfg = deepcopy(self.cfg)
        cfg.update(few_shot=2, few_shot_seed=1, rotation_aug=False)
        model = MODELS.build(cfg)
        model.train()

        for index in range(4):
            inputs = torch.randn(1, 3, 56, 56)
            data_samples = _make_data_samples(1, start_index=index)
            model(inputs, data_samples, mode='loss')

        model.build_memory_bank()

        # seed=1, few_shot=2 -> keep sorted paths [2, 3], each with 16 patch tokens
        assert tuple(model.memory_bank.shape) == (32, 384)

    def test_build_memory_bank_dataloader_collects_only_selected_support(self):
        cfg = deepcopy(self.cfg)
        cfg.update(few_shot=2, few_shot_seed=1, rotation_aug=False)
        model = MODELS.build(cfg)

        dataset = _FakeFewShotDataset(num_samples=4)
        loader = _FakeLoader(dataset)

        def _fake_collect_reference_features(inputs, data_samples):
            sample = data_samples[0]
            model._collected_refs.append(dict(
                img_path=sample.img_path,
                cls_name=sample.cls_name,
                tokens=torch.zeros(16, 384),
                spatial_h=4,
                spatial_w=4,
            ))

        model._collect_reference_features = _fake_collect_reference_features
        model.build_memory_bank(loader)

        assert dataset.requested_indices == [2, 3]
        assert tuple(model.memory_bank.shape) == (32, 384)

    def test_strict_loss_collects_only_selected_support(self):
        cfg = deepcopy(self.cfg)
        cfg.update(preprocess='agnostic', few_shot=2, few_shot_seed=1, rotation_aug=False)
        model = MODELS.build(cfg)

        collected_paths = []

        def _fake_collect_reference_features(inputs, data_samples):
            collected_paths.extend(sample.img_path for sample in data_samples)

        model._collect_reference_features = _fake_collect_reference_features

        for index in range(4):
            inputs = torch.randn(1, 3, 56, 56)
            data_samples = _make_data_samples(1, start_index=index)
            model(inputs, data_samples, mode='loss')

        assert collected_paths == ['/fake/002.png', '/fake/003.png']

    def test_agnostic_rotation_augments_support_bank_with_eight_angles(self):
        cfg = deepcopy(self.cfg)
        cfg.update(preprocess='agnostic', few_shot=None)
        model = MODELS.build(cfg)
        model.train()

        inputs = torch.randn(1, 3, 56, 56)
        model(inputs, _make_data_samples(1), mode='loss')
        model.build_memory_bank()

        # 1 original + 7 extra rotations, 16 tokens each
        assert tuple(model.memory_bank.shape) == (128, 384)

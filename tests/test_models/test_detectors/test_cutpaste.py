"""Tests for CutPasteDetector."""

import math
import random
from unittest import TestCase

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import baoiad  # noqa: F401
from baoiad.models.detectors.cutpaste import CutPasteAugmentation
from baoiad.registry import MODELS
from baoiad.structures import ADDataSample


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


class _InputDataset(Dataset):
    def __init__(self, length=5, H=32, W=32):
        self.length = length
        self.H = H
        self.W = W

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        value = float(idx + 1) / float(self.length + 1)
        return {'inputs': torch.full((3, self.H, self.W), value, dtype=torch.float32)}


class _RepeatDataset(Dataset):
    def __init__(self, dataset, times):
        self.dataset = dataset
        self.times = times

    def __len__(self):
        return len(self.dataset) * self.times

    def __getitem__(self, idx):
        return self.dataset[idx % len(self.dataset)]


@MODELS.register_module(force=True)
class ToyCutPasteBackbone(nn.Module):
    def __init__(self, frozen=True):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False)
        self.out_channels = (8,)
        self.num_features = 12
        self.pre_logits = nn.Linear(8, self.num_features, bias=False)
        if frozen:
            self.eval()
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        return [self.conv(x)]

    def forward_pre_logits(self, x):
        feat = self.forward(x)[0].mean(dim=(2, 3))
        return self.pre_logits(feat)

    def train(self, mode=True):
        if mode and not any(param.requires_grad for param in self.parameters()):
            return super().train(False)
        return super().train(mode)


@MODELS.register_module(force=True)
class ToyCutPasteBNBackbone(nn.Module):
    def __init__(self, frozen=True):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(8)
        self.out_channels = (8,)
        if frozen:
            self.eval()
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        return [self.bn(self.conv(x))]

    def train(self, mode=True):
        if mode and not any(param.requires_grad for param in self.parameters()):
            return super().train(False)
        return super().train(mode)


class TestCutPasteDetector(TestCase):
    def setUp(self):
        self.backbone_cfg = dict(type='ToyCutPasteBackbone', frozen=True)
        self.cfg = dict(
            type='CutPasteDetector',
            backbone=self.backbone_cfg,
            proj_dim=64,
            num_classes=3,
        )

    def test_forward_tensor(self):
        model = MODELS.build(self.cfg)
        model.eval()
        out = model(torch.randn(2, 3, 32, 32), mode='tensor')
        assert out is not None

    def test_forward_loss(self):
        model = MODELS.build(self.cfg)
        model.train()
        data_samples = _make_data_samples(2, 32, 32)
        out = model(torch.randn(2, 3, 32, 32), data_samples, mode='loss')
        assert isinstance(out, dict)

    def test_forward_predict(self):
        model = MODELS.build(self.cfg)
        model.eval()
        data_samples = _make_data_samples(2, 32, 32)
        for _ in range(3):
            model(torch.randn(2, 3, 32, 32), _make_data_samples(2, 32, 32), mode='loss')
        if hasattr(model, 'fit'):
            mock_data = [{'inputs': torch.randn(2, 3, 32, 32)} for _ in range(2)]
            model.fit(mock_data)
        out = model(torch.randn(2, 3, 32, 32), data_samples, mode='predict')
        assert isinstance(out, list)
        assert len(out) == 2
        for sample in out:
            assert math.isfinite(sample.pred_score)
            assert torch.isfinite(sample.pred_anomaly_map).all()
            assert sample.pred_anomaly_map.shape == (1, 32, 32)

    def test_fit_uses_backbone_embeddings_and_unwraps_repeat_dataset(self):
        model = MODELS.build(self.cfg)
        model.eval()

        inner_dataset = _InputDataset(length=5, H=32, W=32)
        loader = DataLoader(_RepeatDataset(inner_dataset, times=4), batch_size=2, shuffle=True)

        model.fit(loader)

        assert model._last_fit_num_samples == len(inner_dataset)
        assert model._gde_mean.shape == (8,)
        assert model._gde_cov_inv.shape == (8, 8)

    def test_set_iter_info_uses_explicit_freeze_iters(self):
        model = MODELS.build(dict(
            type='CutPasteDetector',
            backbone=self.backbone_cfg,
            proj_dim=64,
            num_classes=3,
            freeze_iters=20,
        ))

        assert not any(param.requires_grad for param in model.backbone.parameters())

        model.set_iter_info(19, 256)
        assert not any(param.requires_grad for param in model.backbone.parameters())

        model.set_iter_info(20, 256)
        assert all(param.requires_grad for param in model.backbone.parameters())

    def test_augmentations_preserve_shape_and_range(self):
        random.seed(0)
        torch.manual_seed(0)
        img = torch.rand(3, 32, 32)

        normal_aug = CutPasteAugmentation(
            area_ratio=(0.3, 0.3),
            aspect_ratio=1.0,
            color_jitter=False,
        )
        scar_aug = CutPasteAugmentation(
            scar_width=(4, 4),
            scar_length=(20, 20),
            scar_rotation=(45, 45),
            color_jitter=False,
        )

        normal = normal_aug.cutpaste(img)
        scar = scar_aug.cutpaste_scar(img)

        for output in [normal, scar]:
            assert output.shape == img.shape
            assert torch.isfinite(output).all()
            assert output.min() >= 0.0
            assert output.max() <= 1.0

        assert not torch.allclose(normal, img)
        assert not torch.allclose(scar, img)

    def test_build_augmented_views_returns_consistent_batch(self):
        model = MODELS.build(self.cfg)
        inputs = torch.randn(2, 3, 32, 32)

        augmented = model.build_augmented_views(inputs)

        assert augmented['inputs_01'].shape == (2, 3, 32, 32)
        assert augmented['normal_01'].shape == (2, 3, 32, 32)
        assert augmented['cutpaste_01'].shape == (2, 3, 32, 32)
        assert augmented['scar_01'].shape == (2, 3, 32, 32)
        assert augmented['all_imgs'].shape == (6, 3, 32, 32)
        assert augmented['labels'].tolist() == [0, 0, 1, 1, 2, 2]

        for key in ['inputs_01', 'normal_01', 'cutpaste_01', 'scar_01']:
            tensor = augmented[key]
            assert torch.isfinite(tensor).all()
            assert tensor.min() >= 0.0
            assert tensor.max() <= 1.0

    def test_scar_boost_meets_min_changed_ratio(self):
        random.seed(0)
        torch.manual_seed(0)
        img = torch.rand(3, 64, 64)
        aug = CutPasteAugmentation(
            scar_width=(6, 24),
            scar_length=(16, 48),
            scar_rotation=(-45, 45),
            scar_min_changed_ratio=0.005,
            scar_max_attempts=8,
            color_jitter=False,
        )

        scar = aug.cutpaste_scar(img)
        diff = (scar - img).abs().mean(dim=0)
        changed_ratio = float((diff > 0.05).float().mean().item())
        assert changed_ratio >= 0.005

    def test_fit_gaussian_density_supports_head_embeddings(self):
        model = MODELS.build(self.cfg)
        model.eval()

        inner_dataset = _InputDataset(length=5, H=32, W=32)
        loader = DataLoader(_RepeatDataset(inner_dataset, times=4), batch_size=2, shuffle=True)

        stats = model.fit_gaussian_density(loader, embedding_type='head')

        assert stats['num_samples'] == len(inner_dataset)
        assert stats['embedding_type'] == 'head'
        assert stats['mean'].shape == (128,)
        assert stats['cov_inv'].shape == (128, 128)

    def test_pre_logits_embedding_source_uses_backbone_pre_logits(self):
        model = MODELS.build(dict(
            type='CutPasteDetector',
            backbone=self.backbone_cfg,
            proj_dim=64,
            num_classes=3,
            embedding_source='pre_logits',
        ))
        model.eval()

        loader = [{'inputs': torch.randn(2, 3, 32, 32)} for _ in range(2)]
        model.fit(loader)
        inputs = torch.randn(2, 3, 32, 32)
        embeddings = model.extract_backbone_embedding(inputs)
        scores = model.score_with_backbone_mahalanobis(inputs)

        assert embeddings.shape == (2, 12)
        assert model._gde_mean.shape == (12,)
        assert model._gde_cov_inv.shape == (12, 12)
        assert tuple(scores.shape) == (2,)
        assert torch.isfinite(scores).all()

    def test_train_and_density_embedding_sources_can_differ(self):
        model = MODELS.build(dict(
            type='CutPasteDetector',
            backbone=self.backbone_cfg,
            proj_dim=64,
            num_classes=3,
            train_embedding_source='features_only',
            density_embedding_source='pre_logits',
        ))
        model.eval()

        loader = [{'inputs': torch.randn(2, 3, 32, 32)} for _ in range(2)]
        model.fit(loader)
        inputs = torch.randn(2, 3, 32, 32)

        train_embeddings = model.extract_train_backbone_embedding(inputs)
        density_embeddings = model.extract_density_backbone_embedding(inputs)
        head_embeddings = model.extract_head_embedding(inputs)

        assert train_embeddings.shape == (2, 8)
        assert density_embeddings.shape == (2, 12)
        assert head_embeddings.shape == (2, 128)
        assert model._gde_mean.shape == (12,)
        assert model._gde_cov_inv.shape == (12, 12)
        assert torch.isfinite(model.score_with_backbone_mahalanobis(inputs)).all()

    def test_scoring_helpers_return_expected_shape(self):
        model = MODELS.build(self.cfg)
        model.eval()

        loader = [{'inputs': torch.randn(2, 3, 32, 32)} for _ in range(2)]
        model.fit(loader)
        head_stats = model.fit_gaussian_density(loader, embedding_type='head')
        inputs = torch.randn(2, 3, 32, 32)

        backbone_scores = model.score_with_backbone_mahalanobis(inputs)
        head_scores = model.score_with_head_mahalanobis(inputs, head_stats['mean'], head_stats['cov_inv'])
        classifier_scores = model.score_with_classifier_prob(inputs)

        for scores in [backbone_scores, head_scores, classifier_scores]:
            assert tuple(scores.shape) == (2,)
            assert torch.isfinite(scores).all()

    def test_resnet18_strict_backbone_uses_512d_pooled_embeddings(self):
        model = MODELS.build(dict(
            type='CutPasteDetector',
            backbone=dict(
                type='RawBackbone',
                backbone_name='resnet18',
                pretrained=False,
                frozen=True,
            ),
            num_classes=3,
            head_dims=(512, 128),
            freeze_iters=20,
        ))
        model.eval()

        inputs = torch.randn(2, 3, 64, 64)
        train_embeddings = model.extract_train_backbone_embedding(inputs)
        density_embeddings = model.extract_density_backbone_embedding(inputs)

        loader = [{'inputs': torch.randn(2, 3, 64, 64)} for _ in range(2)]
        model.fit(loader)

        assert train_embeddings.shape == (2, 512)
        assert density_embeddings.shape == (2, 512)
        assert model._gde_mean.shape == (512,)
        assert model._gde_cov_inv.shape == (512, 512)

    def test_keep_backbone_bn_eval_preserves_bn_eval_after_unfreeze(self):
        model = MODELS.build(dict(
            type='CutPasteDetector',
            backbone=dict(type='ToyCutPasteBNBackbone', frozen=True),
            num_classes=3,
            freeze_iters=0,
            keep_backbone_bn_eval=True,
        ))

        model.set_iter_info(0, 256)
        model.train()

        assert model._backbone_unfrozen is True
        assert model.backbone.conv.training is True
        assert model.backbone.bn.training is False

    def test_can_keep_backbone_bn_training_while_backbone_is_frozen(self):
        model = MODELS.build(dict(
            type='CutPasteDetector',
            backbone=dict(type='ToyCutPasteBNBackbone', frozen=True),
            num_classes=3,
            freeze_iters=999,
            force_backbone_eval_while_frozen=False,
        ))

        model.train()

        assert model._backbone_unfrozen is False
        assert model.backbone.conv.training is True
        assert model.backbone.bn.training is True

    def test_strict_frozen_backbone_defaults_to_bn_training_semantics(self):
        model = MODELS.build(dict(
            type='CutPasteDetector',
            backbone=dict(type='ToyCutPasteBNBackbone', frozen=True),
            num_classes=3,
            head_dims=(8, 4),
            freeze_iters=20,
            force_backbone_eval_while_frozen=False,
        ))

        model.train()

        assert model._backbone_unfrozen is False
        assert model.backbone.conv.training is True
        assert model.backbone.bn.training is True

    def test_can_stop_backbone_grad_until_unfreeze(self):
        model = MODELS.build(dict(
            type='CutPasteDetector',
            backbone=dict(type='ToyCutPasteBackbone', frozen=False),
            num_classes=3,
            freeze_iters=5,
            stop_grad_backbone_while_frozen=True,
        ))

        model.train()
        inputs = torch.randn(2, 3, 32, 32)
        data_samples = _make_data_samples(2, 32, 32)

        loss = model(inputs, data_samples, mode='loss')['loss']
        loss.backward()

        assert model.backbone.conv.weight.grad is None
        assert model.classifier.weight.grad is not None

        model.zero_grad(set_to_none=True)
        model.set_iter_info(5, 256)

        loss = model(inputs, data_samples, mode='loss')['loss']
        loss.backward()

        assert model.backbone.conv.weight.grad is not None

"""Tests for GLASSDetector."""

import torch
import torch.nn.functional as F
from mmengine.optim import OptimWrapper, OptimWrapperDict
from unittest import TestCase
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


def _make_strict_data_samples(batch_size, H=64, W=64, feat_size=8):
    samples = []
    for i in range(batch_size):
        s = ADDataSample()
        s.gt_label = 0
        s.gt_mask = torch.zeros(H, W)
        s.set_metainfo({
            'cls_name': 'bottle',
            'img_path': f'/fake/{i}.png',
            'defect_type': 'good',
            'aug': torch.randn(3, H, W),
            'mask_s': F.pad(torch.ones(feat_size - 2, feat_size - 2), (1, 1, 1, 1)),
        })
        samples.append(s)
    return samples


class _FakeGLASSDataset:
    cls_names = ['bottle']
    dataset_name = 'mvtec'
    distribution = 3
    distribution_meta_path = None


class _FakeGLASSLoader:
    def __init__(self, batches):
        self.dataset = _FakeGLASSDataset()
        self._batches = batches

    def __iter__(self):
        return iter(self._batches)


class TestGLASSDetector(TestCase):
    def setUp(self):
        # Use smaller network for fast CPU testing
        self.cfg = dict(
            type='GLASSDetector',
            backbone='resnet18',
            dsc_hidden=64,
            target_dim=256,
            pretrain_embed_dim=256,
            dtd_path=None,  # Skip DTD download for tests
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
        assert out['loss'].item() >= 0

    def test_forward_predict(self):
        model = MODELS.build(self.cfg)
        model.eval()
        data_samples = _make_data_samples(2, 64, 64)
        out = model(torch.randn(2, 3, 64, 64), data_samples, mode='predict')
        assert isinstance(out, list)
        assert len(out) == 2

    def test_predict_scores_in_range(self):
        """Predict mode uses sigmoid, so scores must be in [0, 1]."""
        model = MODELS.build(self.cfg)
        model.eval()
        data_samples = _make_data_samples(2, 64, 64)
        out = model(torch.randn(2, 3, 64, 64), data_samples, mode='predict')
        for sample in out:
            assert 0 <= sample.pred_score <= 1
            assert (sample.pred_anomaly_map >= 0).all()
            assert (sample.pred_anomaly_map <= 1).all()

    def test_gas_branch_produces_anomalies(self):
        """GAS branch should produce features that discriminator scores higher."""
        model = MODELS.build(self.cfg)
        model.train()

        # Extract features
        inputs = torch.randn(2, 3, 64, 64)
        feats, _, _, _ = model.extract_features(inputs)
        if model.pre_proj > 0:
            feats = model.projection(feats)

        # Run GAS synthesis
        gas_feats = model._gas_anomaly_synthesis(feats)

        # Discriminator should score GAS features higher (more anomalous)
        with torch.no_grad():
            normal_scores = torch.sigmoid(model.discriminator(feats))
            gas_scores = torch.sigmoid(model.discriminator(gas_feats))

        # GAS scores should generally be higher (pushed toward anomaly)
        assert gas_scores.mean() > normal_scores.mean()

    def test_backbone_frozen(self):
        """Feature extractor should stay in eval mode during training."""
        model = MODELS.build(self.cfg)
        model.train()
        assert not model.feature_extractor.training

    def test_las_branch_with_no_dtd(self):
        """LAS branch should work with random noise when DTD is unavailable."""
        cfg = dict(
            type='GLASSDetector',
            backbone='resnet18',
            dsc_hidden=64,
            target_dim=256,
            pretrain_embed_dim=256,
            dtd_path=None,
            anomaly_ratio=1.0,  # Always generate anomalies
        )
        model = MODELS.build(cfg)
        model.train()

        inputs = torch.randn(2, 3, 64, 64)
        las_imgs, las_masks = model._generate_las_batch(inputs)

        # Should have generated some masks
        assert las_masks.shape == (2, 64, 64)

    def test_strict_prepare_epoch_and_loss(self):
        cfg = dict(
            type='GLASSDetector',
            strict=True,
            backbone='resnet18',
            dsc_hidden=64,
            target_dim=256,
            pretrain_embed_dim=256,
            distribution=3,
            limit=4,
            image_size=64,
        )
        model = MODELS.build(cfg)
        model.train()

        inputs = torch.randn(2, 3, 64, 64)
        data_samples = _make_strict_data_samples(2, 64, 64, feat_size=8)
        loader = _FakeGLASSLoader([dict(inputs=inputs.clone(), data_samples=data_samples)])
        model.prepare_strict_epoch(loader)

        assert model.strict_center is not None
        assert model.svd == 1

        losses = model(inputs, data_samples, mode='loss')
        assert losses['loss'].item() >= 0
        assert torch.isfinite(losses['loss'])

    def test_strict_predict_scores_in_range(self):
        cfg = dict(
            type='GLASSDetector',
            strict=True,
            backbone='resnet18',
            dsc_hidden=64,
            target_dim=256,
            pretrain_embed_dim=256,
            distribution=3,
            image_size=64,
        )
        model = MODELS.build(cfg)
        model.eval()
        data_samples = _make_data_samples(2, 64, 64)
        out = model(torch.randn(2, 3, 64, 64), data_samples, mode='predict')
        assert len(out) == 2
        for sample in out:
            assert 0 <= sample.pred_score <= 1
            assert torch.isfinite(sample.pred_anomaly_map).all()

    def test_strict_train_step_with_multi_optimizer(self):
        cfg = dict(
            type='GLASSDetector',
            strict=True,
            backbone='resnet18',
            dsc_hidden=64,
            target_dim=256,
            pretrain_embed_dim=256,
            distribution=3,
            limit=4,
            image_size=64,
        )
        model = MODELS.build(cfg)
        model.train()

        inputs = torch.randn(2, 3, 64, 64)
        data_samples = _make_strict_data_samples(2, 64, 64, feat_size=8)
        loader = _FakeGLASSLoader([dict(inputs=inputs.clone(), data_samples=data_samples)])
        model.prepare_strict_epoch(loader)

        optim_wrapper = OptimWrapperDict(
            projection=OptimWrapper(torch.optim.Adam(model.projection.parameters(), lr=1e-4, weight_decay=1e-5)),
            discriminator=OptimWrapper(torch.optim.AdamW(model.discriminator.parameters(), lr=2e-4, weight_decay=1e-5)),
        )

        outputs = model.train_step(dict(inputs=inputs, data_samples=data_samples), optim_wrapper)
        assert 'loss' in outputs
        assert torch.isfinite(outputs['loss'])

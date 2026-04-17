"""Tests for ComposeAD detector and ScoringHeads."""
import pytest
import torch
import numpy as np

import baoiad  # trigger registry
from baoiad.registry import MODELS
from baoiad.structures import ADDataSample


def _make_dummy_backbone_config(out_indices=(2, 3)):
    return dict(
        type='TIMMBackbone',
        model_name='resnet18',
        pretrained=False,
        features_only=True,
        out_indices=out_indices,
        frozen=True,
    )


def _make_dummy_data_samples(B=2):
    samples = []
    for i in range(B):
        s = ADDataSample()
        s.gt_label = 0
        s.cls_name = 'bottle'
        s.set_metainfo({'img_path': f'test/{i}.png'})
        samples.append(s)
    return samples


class TestComposeAD:
    """Test ComposeAD detector."""

    def test_build_knn(self):
        model = MODELS.build(dict(
            type='ComposeAD',
            backbone=_make_dummy_backbone_config(),
            neck=dict(type='MultiScalePooling', output_size=7),
            scoring_head=dict(
                type='KNNScoringHead',
                coreset_ratio=0.1,
                num_neighbors=3,
                input_size=(256, 256),
                blur_sigma=4.0,
            ),
            freeze_backbone=True,
        ))
        assert model.scoring_head is not None
        assert model.backbone is not None

    def test_build_gaussian(self):
        model = MODELS.build(dict(
            type='ComposeAD',
            backbone=_make_dummy_backbone_config(out_indices=(1, 2, 3)),
            neck=None,
            scoring_head=dict(
                type='GaussianScoringHead',
                d_reduced=100,
                input_size=(256, 256),
                blur_sigma=4.0,
            ),
            freeze_backbone=True,
        ))
        assert model.scoring_head is not None

    def test_build_pca(self):
        model = MODELS.build(dict(
            type='ComposeAD',
            backbone=_make_dummy_backbone_config(out_indices=(3,)),
            neck=None,
            scoring_head=dict(
                type='PCAScoringHead',
                pca_level=0.97,
                scoring='fre',
                input_size=(256, 256),
            ),
            freeze_backbone=True,
        ))
        assert model.scoring_head is not None

    def test_knn_loss_then_predict(self):
        model = MODELS.build(dict(
            type='ComposeAD',
            backbone=_make_dummy_backbone_config(),
            neck=dict(type='MultiScalePooling', output_size=7),
            scoring_head=dict(
                type='KNNScoringHead',
                coreset_ratio=0.5,
                num_neighbors=1,
                patchsize=1,
                input_size=(256, 256),
                blur_sigma=0.0,
            ),
            freeze_backbone=True,
        ))
        model.eval()
        x = torch.randn(2, 3, 64, 64)
        samples = _make_dummy_data_samples(2)

        # Training: collect features
        for _ in range(3):
            result = model(x, samples, mode='loss')
            assert 'loss' in result

        # Fit
        model.build_memory_bank()
        assert model.scoring_head.memory_bank is not None

        # Predict
        preds = model(x, samples, mode='predict')
        assert len(preds) == 2
        assert hasattr(preds[0], 'pred_score')
        assert hasattr(preds[0], 'pred_anomaly_map')
        assert isinstance(preds[0].pred_score, float)
        assert preds[0].pred_anomaly_map.shape[-2:] == (256, 256)

    def test_gaussian_loss_fit_predict(self):
        model = MODELS.build(dict(
            type='ComposeAD',
            backbone=_make_dummy_backbone_config(out_indices=(1, 2, 3)),
            neck=None,
            scoring_head=dict(
                type='GaussianScoringHead',
                d_reduced=50,
                eps=0.01,
                input_size=(64, 64),
                blur_sigma=0.0,
            ),
            freeze_backbone=True,
        ))
        model.eval()
        x = torch.randn(4, 3, 64, 64)
        samples = _make_dummy_data_samples(4)

        # Training
        for _ in range(3):
            result = model(x, samples, mode='loss')
            assert 'loss' in result

        # Fit
        model.build_memory_bank()
        assert model.scoring_head.mean is not None
        assert model.scoring_head.cov_inv is not None

        # Predict
        preds = model(x, samples, mode='predict')
        assert len(preds) == 4
        assert hasattr(preds[0], 'pred_score')
        assert preds[0].pred_anomaly_map.shape[-2:] == (64, 64)

    def test_pca_loss_fit_predict(self):
        model = MODELS.build(dict(
            type='ComposeAD',
            backbone=_make_dummy_backbone_config(out_indices=(3,)),
            neck=None,
            scoring_head=dict(
                type='PCAScoringHead',
                pca_level=0.97,
                scoring='fre',
                pooling_kernel_size=2,
                input_size=(64, 64),
                blur_sigma=0.0,
            ),
            freeze_backbone=True,
        ))
        model.eval()
        x = torch.randn(4, 3, 64, 64)
        samples = _make_dummy_data_samples(4)

        # Training
        for _ in range(3):
            result = model(x, samples, mode='loss')
            assert 'loss' in result

        # Fit
        model.build_memory_bank()
        assert model.scoring_head.singular_vectors.numel() > 0

        # Predict
        preds = model(x, samples, mode='predict')
        assert len(preds) == 4
        assert hasattr(preds[0], 'pred_score')

    def test_tensor_mode(self):
        model = MODELS.build(dict(
            type='ComposeAD',
            backbone=_make_dummy_backbone_config(),
            neck=dict(type='MultiScalePooling', output_size=7),
            scoring_head=dict(
                type='KNNScoringHead',
                input_size=(64, 64),
            ),
            freeze_backbone=True,
        ))
        model.eval()
        x = torch.randn(2, 3, 64, 64)
        feats = model(x, mode='tensor')
        assert isinstance(feats, tuple)
        assert len(feats) > 0

    def test_build_memory_bank_delegates(self):
        """build_memory_bank calls scoring_head.fit()."""
        model = MODELS.build(dict(
            type='ComposeAD',
            backbone=_make_dummy_backbone_config(),
            neck=dict(type='MultiScalePooling', output_size=7),
            scoring_head=dict(
                type='KNNScoringHead',
                input_size=(64, 64),
            ),
            freeze_backbone=True,
        ))
        model.eval()
        x = torch.randn(2, 3, 64, 64)
        samples = _make_dummy_data_samples(2)
        model(x, samples, mode='loss')  # collect features
        model.build_memory_bank()
        assert model.scoring_head.memory_bank is not None


class TestKNNScoringHead:
    """Test KNNScoringHead independently."""

    def test_coreset_selection(self):
        head = MODELS.build(dict(
            type='KNNScoringHead',
            coreset_ratio=0.5,
            feature_selection='coreset',
            num_neighbors=1,
            patchsize=1,
            input_size=(32, 32),
            blur_sigma=0.0,
        ))
        # Simulate collecting features
        feats = (torch.randn(2, 64, 7, 7), torch.randn(2, 128, 7, 7))
        head.loss(feats)
        head.fit()
        assert head.memory_bank is not None
        assert head._nn_index is not None
        # Should be ~50% of patches
        n_original = 2 * 7 * 7  # B * H * W
        assert head.memory_bank.shape[0] <= n_original

    def test_full_selection(self):
        head = MODELS.build(dict(
            type='KNNScoringHead',
            coreset_ratio=1.0,
            feature_selection='full',
            num_neighbors=1,
            patchsize=1,
            input_size=(32, 32),
            blur_sigma=0.0,
        ))
        feats = (torch.randn(2, 64, 7, 7),)
        head.loss(feats)
        head.fit()
        assert head.memory_bank is not None


class TestGaussianScoringHead:
    """Test GaussianScoringHead independently."""

    def test_random_dim_selection(self):
        head = MODELS.build(dict(
            type='GaussianScoringHead',
            d_reduced=50,
            dim_reduction='random',
            input_size=(32, 32),
            blur_sigma=0.0,
        ))
        feats = (torch.randn(4, 64, 8, 8), torch.randn(4, 128, 8, 8), torch.randn(4, 256, 8, 8))
        for _ in range(3):
            head.loss(feats)
        head.fit()
        assert head.mean.shape[0] == 64  # 8*8 positions
        assert head.cov_inv.shape == (64, 50, 50)

    def test_predict_after_fit(self):
        head = MODELS.build(dict(
            type='GaussianScoringHead',
            d_reduced=30,
            input_size=(32, 32),
            blur_sigma=0.0,
        ))
        feats = (torch.randn(4, 64, 8, 8), torch.randn(4, 128, 8, 8))
        for _ in range(3):
            head.loss(feats)
        head.fit()
        preds = head.predict(feats)
        assert len(preds) == 4
        assert all(isinstance(p.pred_score, float) for p in preds)


class TestPCAScoringHead:
    """Test PCAScoringHead independently."""

    def test_pca_fit(self):
        head = MODELS.build(dict(
            type='PCAScoringHead',
            pca_level=0.95,
            scoring='fre',
            pooling_kernel_size=2,
            input_size=(32, 32),
        ))
        feats = (torch.randn(4, 256, 8, 8),)
        for _ in range(3):
            head.loss(feats)
        head.fit()
        assert head.singular_vectors.shape[1] > 0
        assert head.singular_values.shape[0] > 0

    def test_nll_scoring(self):
        head = MODELS.build(dict(
            type='PCAScoringHead',
            pca_level=0.95,
            scoring='nll',
            pooling_kernel_size=2,
            input_size=(32, 32),
        ))
        feats = (torch.randn(4, 256, 8, 8),)
        for _ in range(3):
            head.loss(feats)
        head.fit()
        preds = head.predict(feats)
        assert len(preds) == 4

    def test_coreset_selection(self):
        head = MODELS.build(dict(
            type='PCAScoringHead',
            feature_selection='coreset',
            coreset_ratio=0.5,
            pca_level=0.95,
            scoring='fre',
            pooling_kernel_size=2,
            input_size=(32, 32),
        ))
        feats = (torch.randn(4, 256, 8, 8),)
        for _ in range(3):
            head.loss(feats)
        head.fit()
        assert head.singular_vectors.numel() > 0

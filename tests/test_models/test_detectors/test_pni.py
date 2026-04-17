"""Tests for PNI detector."""

import pytest
import torch
import numpy as np

import baoiad  # noqa: F401

from baoiad.models.detectors.pni import PNI
from baoiad.models.heads.pni_head import PNIHead

# Reuse dummy backbone from test_base_ad_model
from tests.test_models.test_base_ad_model import _DummyBackbone  # noqa: F401


# Register a simple neck for testing
import torch.nn as nn
from baoiad.registry import MODELS


@MODELS.register_module()
class _DummyNeck(nn.Module):
    """Dummy neck that returns input features unchanged."""

    def __init__(self):
        super().__init__()

    def forward(self, feats):
        return feats


@MODELS.register_module()
class _DummyPNIBackbone(nn.Module):
    """Dummy backbone with multi-scale outputs for PNI testing."""

    def __init__(self, out_channels=64):
        super().__init__()
        self.conv = nn.Conv2d(3, out_channels, 1)

    def forward(self, x):
        feat = self.conv(x)
        return (feat, feat)  # Return two scales


_CFG = dict(
    backbone=dict(type='_DummyPNIBackbone', out_channels=32),
    neck=dict(type='_DummyNeck'),
    head=dict(
        type='PNIHead',
        coreset_ratio=0.5,  # High ratio for small test
        distribution_size=16,  # Small for testing
        neighborhood_size=3,
        mlp_layers=2,
        mlp_channels=32,
        temperature=1.0,
        lambda_param=1.0,
        num_neighbors=1,
        input_size=(32, 32),
        mlp_epochs=1,  # Just 1 epoch for testing
        mlp_lr=1e-3,
    ),
)


class TestPNI:
    def _build(self):
        return PNI(**_CFG)

    def test_forward_tensor(self):
        model = self._build()
        out = model(torch.randn(2, 3, 32, 32), mode='tensor')
        assert isinstance(out, tuple)
        assert len(out) == 2  # Two scales

    def test_forward_loss(self):
        model = self._build()
        out = model(torch.randn(2, 3, 32, 32), mode='loss')
        assert isinstance(out, dict)
        assert 'loss' in out

    def test_forward_predict_no_memory_bank(self):
        """Predict should fail before memory bank is built."""
        model = self._build()
        with pytest.raises(AssertionError, match='Memory bank not built'):
            model(torch.randn(1, 3, 32, 32), mode='predict')

    def test_backbone_frozen(self):
        model = self._build()
        for p in model.backbone.parameters():
            assert not p.requires_grad

    def test_invalid_mode(self):
        model = self._build()
        with pytest.raises(RuntimeError):
            model(torch.randn(1, 3, 32, 32), mode='bad')

    def test_build_memory_bank_and_predict(self):
        """Test full flow: collect features, build memory bank, predict."""
        model = self._build()

        # Simulate training: collect features
        for _ in range(3):
            model(torch.randn(2, 3, 32, 32), mode='loss')

        # Build memory bank
        model.build_memory_bank()
        assert model.head.mlp is not None
        assert model.head.last_build_info['embedding_coreset_shape'][1] == 1024
        assert model.head.last_build_info['mlp_fit_info']['train_size'] > 0

        # Now predict should work
        results = model(torch.randn(1, 3, 32, 32), mode='predict')
        assert isinstance(results, list)
        assert len(results) == 1
        assert hasattr(results[0], 'pred_score')
        assert hasattr(results[0], 'pred_anomaly_map')
        assert results[0].pred_score >= 0
        assert model.head.last_predict_summary is None

        feats = model(torch.randn(1, 3, 32, 32), mode='tensor')
        debug = model.head.debug_predict_summary(feats)
        assert 'coor_mask_rate' in debug
        assert 'nb_mask_rate' in debug
        assert debug['nb_threshold'] == pytest.approx(
            model.head.softmax_nb_gamma / len(model.head.embedding_coreset)
        )
        assert 'anomaly_map_raw_max' in debug
        assert model.head.last_scoring_summary is not None

    def test_pni_head_init(self):
        """Test PNIHead initialization."""
        head = PNIHead(
            coreset_ratio=0.1,
            distribution_size=128,
            neighborhood_size=5,
            mlp_layers=3,
            mlp_channels=64,
        )
        assert head.coreset_ratio == 0.1
        assert head.distribution_size == 128
        assert head.neighborhood_size == 5
        assert head.mlp_layers == 3
        assert head.mlp_channels == 64
        assert head.mlp_batch_size == 2048
        assert head.max_train_samples == 0
        assert head.approximate_coreset is False
        assert head.candidate_neighbors == 100


class TestPNIHead:
    def test_init_defaults(self):
        head = PNIHead()
        assert head.coreset_ratio == 0.01
        assert head.distribution_size == 2048
        assert head.neighborhood_size == 9
        assert head.mlp_layers == 10
        assert head.mlp_channels == 2048
        assert head.mlp_batch_size == 2048
        assert head.max_train_samples == 0
        assert head.approximate_coreset is False

    def test_collect_features(self):
        head = PNIHead(distribution_size=16, mlp_layers=2, mlp_channels=32)
        feats = (torch.randn(2, 32, 8, 8), torch.randn(2, 32, 8, 8))
        head.collect_features(feats)
        assert len(head._train_features) == 1
        assert head._spatial_shape == (8, 8)
        assert head._valid_spatial_shape == (4, 4)
        assert head._feature_dim == 1024

    def test_coordinate_model_keeps_full_grid_rows(self):
        head = PNIHead(
            coreset_ratio=1.0,
            distribution_size=32,
            mlp_layers=2,
            mlp_channels=32,
            mlp_epochs=1,
            max_train_samples=16,
        )
        feats = (torch.randn(1, 8, 8, 8), torch.randn(1, 8, 4, 4))
        head.collect_features(feats)
        head.build_memory_bank()

        assert head.embedding_coreset.shape[0] == 16
        assert head.embedding_coreset_with_edge.shape[0] == 64
        assert head.coor_model.shape[0] == 64
        assert head.last_build_info['mlp_fit_info']['train_size'] == 15
        assert head.last_build_info['mlp_fit_info']['val_size'] == 1
        assert 'coor_without_edge_s' in head.last_build_info['stage_times']

    def test_rank_aligned_mask_gathers_by_neighbor_rank(self):
        mask_by_id = np.array([
            [10, 11, 12, 13],
            [20, 21, 22, 23],
        ], dtype=np.int64)
        ranked_indices = np.array([
            [2, 0, 3],
            [1, 3, 0],
        ], dtype=np.int64)

        gathered = PNIHead._gather_rank_aligned_mask(mask_by_id, ranked_indices)

        expected = np.array([
            [12, 10, 13],
            [21, 23, 20],
        ], dtype=np.int64)
        assert np.array_equal(gathered, expected)

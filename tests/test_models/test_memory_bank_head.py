"""Tests for MemoryBankHead."""

import numpy as np
import pytest
import torch
from baoiad.structures import ADDataSample

import baoiad  # noqa: F401

from baoiad.models.heads.memory_bank_head import MemoryBankHead


@pytest.fixture
def head():
    return MemoryBankHead(
        coreset_ratio=0.5,
        num_neighbors=1,
        patchsize=1,
        blur_sigma=0.0,
    )


@pytest.fixture
def dummy_feats():
    """Two feature maps simulating multi-scale output: (B=2, C, H=4, W=4)."""
    return (torch.randn(2, 32, 4, 4), torch.randn(2, 64, 4, 4))


@pytest.fixture
def shifted_feats(dummy_feats):
    """Prediction features that differ from the collected support bank."""
    shifted = []
    for feat in dummy_feats:
        offset = torch.linspace(0.0, 1.0, feat.numel(), dtype=feat.dtype).reshape_as(feat)
        shifted.append(feat + 0.1 * offset)
    return tuple(shifted)


class TestMemoryBankHead:
    def test_collect_features(self, head, dummy_feats):
        head.collect_features(dummy_feats)
        assert len(head._train_features) == 1
        # 2 images * 4*4 patches = 32 patches, dim=96
        assert head._train_features[0].shape == (32, 96)

    def test_build_memory_bank(self, head, dummy_feats):
        head.collect_features(dummy_feats)
        head.build_memory_bank()
        assert head.memory_bank is not None
        assert head._nn_index is not None
        assert head.memory_bank.shape[1] == 96
        # coreset_ratio=0.5 → ~16 samples
        assert head.memory_bank.shape[0] == 16

    def test_predict(self, head, dummy_feats):
        # Collect and build
        head.collect_features(dummy_feats)
        head.build_memory_bank()

        # Predict
        samples = [ADDataSample() for _ in range(2)]
        results = head.predict(dummy_feats, samples)
        assert len(results) == 2
        for r in results:
            assert hasattr(r, 'pred_score')
            assert hasattr(r, 'pred_anomaly_map')
            assert r.pred_anomaly_map.shape == (1, 4, 4)

    def test_predict_uses_input_size_for_full_resolution_map(self, dummy_feats, shifted_feats):
        head = MemoryBankHead(
            coreset_ratio=0.5,
            num_neighbors=1,
            patchsize=1,
            input_size=(8, 8),
            blur_sigma=0.0,
        )
        head.collect_features(dummy_feats)
        head.build_memory_bank()

        results = head.predict(shifted_feats, [ADDataSample() for _ in range(2)])
        for result in results:
            assert result.pred_anomaly_map.shape == (1, 8, 8)
            assert torch.isfinite(result.pred_anomaly_map).all()

    def test_predict_blur_preserves_shape_and_changes_map(self, dummy_feats, shifted_feats):
        raw_head = MemoryBankHead(
            coreset_ratio=0.5,
            num_neighbors=1,
            patchsize=1,
            input_size=(8, 8),
            blur_sigma=0.0,
        )
        blur_head = MemoryBankHead(
            coreset_ratio=0.5,
            num_neighbors=1,
            patchsize=1,
            input_size=(8, 8),
            blur_sigma=1.0,
        )
        for test_head in (raw_head, blur_head):
            test_head.collect_features(dummy_feats)
            test_head.build_memory_bank()

        raw_results = raw_head.predict(shifted_feats, [ADDataSample() for _ in range(2)])
        blur_results = blur_head.predict(shifted_feats, [ADDataSample() for _ in range(2)])

        for raw_result, blur_result in zip(raw_results, blur_results):
            assert blur_result.pred_anomaly_map.shape == raw_result.pred_anomaly_map.shape
            assert torch.isfinite(blur_result.pred_anomaly_map).all()
            assert not torch.allclose(blur_result.pred_anomaly_map, raw_result.pred_anomaly_map)

    def test_compute_image_score_softmax_pooling_helper(self, head):
        raw_map = torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=torch.float32)
        flat = raw_map.flatten()
        expected = float((flat * torch.softmax(flat, dim=0)).sum())
        assert head._compute_image_score(raw_map) == pytest.approx(expected)

    def test_reduce_patch_scores_supports_first_and_mean(self):
        head = MemoryBankHead(blur_sigma=0.0)
        distances = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        assert head._reduce_patch_scores(distances)[0] == pytest.approx(1.0)

        head.patch_score_reduction = 'mean'
        assert head._reduce_patch_scores(distances)[0] == pytest.approx(2.0)

    def test_weighted_patchcore_score_uses_support_neighbors(self):
        head = MemoryBankHead(
            num_neighbors=3,
            reweight_scores=True,
            blur_sigma=0.0,
        )
        head.memory_bank = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [3.0, 3.0],
            ],
            dtype=np.float32,
        )

        patch_scores = torch.tensor([[0.2, 0.9]], dtype=torch.float32)
        locations = np.array([[0, 3]], dtype=np.int64)
        patch_embeddings = np.array([[[0.1, 0.0], [2.5, 2.5]]], dtype=np.float32)
        score = head._compute_weighted_patchcore_score(
            patch_scores,
            locations,
            patch_embeddings,
        )
        assert score.shape == (1,)
        assert 0.0 < float(score[0]) < 0.9

    def test_predict_without_reweight_uses_postprocessed_map_max(self, dummy_feats, shifted_feats):
        head = MemoryBankHead(
            coreset_ratio=0.5,
            num_neighbors=1,
            patchsize=1,
            input_size=(8, 8),
            blur_sigma=1.0,
            reweight_scores=False,
        )
        head.collect_features(dummy_feats)
        head.build_memory_bank()

        one_sample_feats = tuple(feat[:1] for feat in shifted_feats)
        result = head.predict(one_sample_feats, [ADDataSample()])[0]
        expected = float(result.pred_anomaly_map.max().item())
        assert result.pred_score == pytest.approx(expected)

    def test_predict_without_reweight_can_use_raw_or_upsampled_source(self, dummy_feats, shifted_feats):
        raw_head = MemoryBankHead(
            coreset_ratio=0.5,
            num_neighbors=1,
            patchsize=1,
            input_size=(8, 8),
            blur_sigma=1.0,
            reweight_scores=False,
            image_score_source='raw',
        )
        upsampled_head = MemoryBankHead(
            coreset_ratio=0.5,
            num_neighbors=1,
            patchsize=1,
            input_size=(8, 8),
            blur_sigma=1.0,
            reweight_scores=False,
            image_score_source='upsampled',
        )
        for test_head in (raw_head, upsampled_head):
            test_head.collect_features(dummy_feats)
            test_head.build_memory_bank()

        one_sample_feats = tuple(feat[:1] for feat in shifted_feats)
        raw_result = raw_head.predict(one_sample_feats, [ADDataSample()])[0]
        upsampled_result = upsampled_head.predict(one_sample_feats, [ADDataSample()])[0]

        assert raw_result.pred_score != pytest.approx(upsampled_result.pred_score)
        assert torch.isfinite(raw_result.pred_anomaly_map).all()
        assert torch.isfinite(upsampled_result.pred_anomaly_map).all()

    def test_loss_returns_dummy(self, head, dummy_feats):
        out = head.loss(dummy_feats)
        assert 'loss' in out
        assert out['loss'].requires_grad

    def test_patchify_aggregation(self):
        head = MemoryBankHead(patchsize=3)
        x = torch.randn(2, 64, 8, 8)
        patches = head._patchify_and_aggregate(x)
        assert patches.shape == (2 * 8 * 8, 64)

    def test_predict_without_build_raises(self, head, dummy_feats):
        samples = [ADDataSample() for _ in range(2)]
        with pytest.raises(AssertionError, match='Memory bank not built'):
            head.predict(dummy_feats, samples)

    def test_coreset_sampling_is_deterministic(self):
        features = np.arange(240, dtype=np.float32).reshape(40, 6)
        first = MemoryBankHead._coreset_sampling(features, 8)
        second = MemoryBankHead._coreset_sampling(features, 8)
        np.testing.assert_allclose(first, second)

    def test_approximate_coreset_sampling_is_deterministic(self):
        head = MemoryBankHead(
            coreset_ratio=0.5,
            num_neighbors=1,
            patchsize=1,
            blur_sigma=0.0,
            coreset_sampling_method='approx_greedy',
            coreset_device='cpu',
        )
        features = np.arange(240, dtype=np.float32).reshape(40, 6)
        first = head._approximate_coreset_sampling(features, 8)
        second = head._approximate_coreset_sampling(features, 8)
        np.testing.assert_allclose(first, second)

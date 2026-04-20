"""Tests for SPADEDetector."""

import pytest
import torch
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

class TestSPADEDetector(TestCase):
    def setUp(self):
        self.cfg = dict(type='SPADEDetector', backbone='resnet18', k=3)

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
        # Memory bank methods need collect phase before predict
        # Run a few forward passes in loss mode to populate memory
        for _ in range(3):
            model(torch.randn(2, 3, 64, 64), _make_data_samples(2, 64, 64), mode='loss')
        if hasattr(model, 'fit'):
            model.fit()
        out = model(torch.randn(2, 3, 64, 64), data_samples, mode='predict')
        assert isinstance(out, list)
        assert len(out) == 2

    def test_predict_raises_before_memory_banks_are_built(self):
        model = MODELS.build(self.cfg)
        model.eval()
        with pytest.raises(RuntimeError, match='memory banks are not built'):
            model(torch.randn(1, 3, 64, 64), _make_data_samples(1, 64, 64), mode='predict')

    def test_feature_caches_round_trip_through_state_dict(self):
        model = MODELS.build(self.cfg)
        model.train()
        model(torch.randn(2, 3, 64, 64), _make_data_samples(2, 64, 64), mode='loss')
        assert model._layer_features[0]
        assert model._gap_features

        restored = MODELS.build(self.cfg)
        restored.load_state_dict(model.state_dict(), strict=False)

        assert len(restored._layer_features[0]) == 1
        assert len(restored._gap_features) == 1
        assert torch.allclose(restored._layer_features[0][0], model._layer_features[0][0])
        assert torch.allclose(restored._gap_features[0], model._gap_features[0])

    def test_knn_kth_distance_matches_full_cdist_when_memory_is_chunked(self):
        model = MODELS.build(dict(**self.cfg, knn_chunk_size=2, knn_memory_chunk_size=3))
        queries = torch.randn(4, 8)
        memory = torch.randn(7, 8)

        actual = model._knn_kth_distance(queries, memory, k=3)

        expected = torch.cdist(queries, memory).topk(3, dim=1, largest=False).values[:, -1]
        assert torch.allclose(actual, expected, atol=1e-6)

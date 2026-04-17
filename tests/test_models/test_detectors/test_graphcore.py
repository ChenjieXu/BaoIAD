"""Tests for GraphCoreDetector."""

from unittest import TestCase

import numpy as np
import pytest
import torch

import baoiad  # noqa: F401
from baoiad.models.detectors.graphcore import _kcenter_greedy
from baoiad.registry import MODELS
from baoiad.structures import ADDataSample


def _make_data_samples(batch_size: int, height: int = 224, width: int = 224):
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


def _graphcore_cfg():
    return dict(
        type='GraphCoreDetector',
        backbone=dict(
            type='GraphCoreViGBackbone',
            model_name='vig_ti_224_gelu',
            pretrained=False,
            frozen=True,
        ),
        n_neighbours=3,
        sampler_percentage=0.05,
        layer_num_1=3,
        layer_num_2=4,
        input_size=(224, 224),
        smoothing_sigma=4.0,
        random_seed=66,
        coreset_initial_index=0,
    )


class TestGraphCoreDetector(TestCase):
    def test_kcenter_greedy_is_deterministic_for_strict_seed_and_start(self):
        features = np.array([[0.0], [5.0], [10.0], [1.0], [9.0]], dtype=np.float32)

        selected = _kcenter_greedy(features, n_select=3, seed=66, initial_index=0)
        repeated = _kcenter_greedy(features, n_select=3, seed=66, initial_index=0)

        assert selected.tolist() == [0, 2, 1]
        np.testing.assert_array_equal(selected, repeated)

    def test_forward_tensor(self):
        model = MODELS.build(_graphcore_cfg())
        model.eval()
        out = model(torch.randn(1, 3, 224, 224), mode='tensor')
        assert out.shape[0] == 1
        assert out.ndim == 4

    def test_forward_loss_collects_embeddings(self):
        model = MODELS.build(_graphcore_cfg())
        model.train()
        out = model(torch.randn(1, 3, 224, 224), _make_data_samples(1), mode='loss')
        assert isinstance(out, dict)
        assert 'loss' in out
        assert len(model._train_embeddings) == 1
        assert model._train_embeddings[0].ndim == 2

    def test_build_memory_bank_from_loss_embeddings(self):
        model = MODELS.build(_graphcore_cfg())
        model.train()
        for _ in range(2):
            model(torch.randn(1, 3, 224, 224), _make_data_samples(1), mode='loss')
        model.build_memory_bank()
        assert model.embedding_coreset is not None
        assert model.embedding_coreset.ndim == 2
        assert model.embedding_coreset.shape[0] >= 1

    def test_forward_predict_outputs_full_res_map(self):
        model = MODELS.build(_graphcore_cfg())
        model.train()
        for _ in range(2):
            model(torch.randn(1, 3, 224, 224), _make_data_samples(1), mode='loss')
        model.build_memory_bank()
        model.eval()

        out = model(torch.randn(1, 3, 224, 224), _make_data_samples(1), mode='predict')
        assert isinstance(out, list)
        assert len(out) == 1
        assert torch.isfinite(torch.tensor(out[0].pred_score))
        assert out[0].pred_anomaly_map.shape == (1, 224, 224)

    def test_embedding_concate_matches_expected_shape(self):
        x = torch.randn(1, 2, 4, 4)
        y = torch.randn(1, 3, 2, 2)
        embedding = MODELS.get('GraphCoreDetector').embedding_concate(x, y)
        assert embedding.shape == (1, 5, 4, 4)

    def test_reduce_image_score_supports_smooth_percentiles(self):
        model = MODELS.build(_graphcore_cfg())
        patch_map = torch.tensor([[0.0, 1.0], [2.0, 4.0]], dtype=torch.float32).numpy()
        smooth_map = torch.tensor([[1.0, 2.0], [3.0, 5.0]], dtype=torch.float32).numpy()

        model.image_score_mode = 'smooth_p95'
        assert model._reduce_image_score(patch_map, smooth_map) == pytest.approx(np.percentile(smooth_map, 95))

        model.image_score_mode = 'raw_max'
        assert model._reduce_image_score(patch_map, smooth_map) == pytest.approx(4.0)

    def test_reduce_image_score_supports_class_overrides(self):
        model = MODELS.build(_graphcore_cfg())
        model.image_score_mode = 'raw_max'
        model.image_score_mode_overrides = {'bottle': 'smooth_p95'}
        patch_map = torch.tensor([[0.0, 1.0], [2.0, 4.0]], dtype=torch.float32).numpy()
        smooth_map = torch.tensor([[1.0, 2.0], [3.0, 5.0]], dtype=torch.float32).numpy()

        sample = ADDataSample()
        sample.cls_name = 'bottle'
        assert model._reduce_image_score(patch_map, smooth_map, sample) == pytest.approx(np.percentile(smooth_map, 95))

    def test_init_rejects_unsupported_image_score_mode(self):
        cfg = _graphcore_cfg()
        cfg['image_score_mode'] = 'bad_mode'

        with pytest.raises(ValueError, match='Unsupported GraphCore image score mode'):
            MODELS.build(cfg)

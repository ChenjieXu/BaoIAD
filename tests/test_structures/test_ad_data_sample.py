"""Tests for ADDataSample and build_predict_results."""

import torch
import pytest

from baoiad.structures import ADDataSample
from baoiad.models.predict_utils import build_predict_results


class TestADDataSample:
    """Test ADDataSample data structure."""

    def test_inherits_base_data_element(self):
        from mmengine.structures import BaseDataElement
        s = ADDataSample()
        assert isinstance(s, BaseDataElement)

    def test_set_get_pred_score(self):
        s = ADDataSample()
        s.pred_score = 0.75
        assert s.pred_score == 0.75

    def test_set_get_gt_label(self):
        s = ADDataSample()
        s.gt_label = 1
        assert s.gt_label == 1

    def test_set_get_gt_mask(self):
        s = ADDataSample()
        mask = torch.ones(256, 256)
        s.gt_mask = mask
        assert torch.equal(s.gt_mask, mask)

    def test_set_get_pred_anomaly_map(self):
        s = ADDataSample()
        amap = torch.randn(1, 64, 64)
        s.pred_anomaly_map = amap
        assert torch.equal(s.pred_anomaly_map, amap)

    def test_set_get_string_fields(self):
        s = ADDataSample()
        s.cls_name = 'bottle'
        s.img_path = '/data/img.png'
        s.defect_type = 'broken'
        assert s.cls_name == 'bottle'
        assert s.img_path == '/data/img.png'
        assert s.defect_type == 'broken'

    def test_multiple_fields_coexist(self):
        s = ADDataSample()
        s.gt_label = 0
        s.pred_score = 0.1
        s.cls_name = 'cable'
        assert s.gt_label == 0
        assert s.pred_score == 0.1
        assert s.cls_name == 'cable'


class TestBuildPredictResults:
    """Test build_predict_results utility."""

    def test_basic_with_scores_and_maps(self):
        scores = torch.tensor([0.1, 0.9])
        maps = torch.randn(2, 1, 64, 64)
        results = build_predict_results(None, scores, maps)
        assert len(results) == 2
        assert isinstance(results[0], ADDataSample)
        assert results[0].pred_score == pytest.approx(0.1, abs=1e-5)
        assert results[1].pred_score == pytest.approx(0.9, abs=1e-5)
        assert results[0].pred_anomaly_map.shape == (1, 64, 64)

    def test_2d_score_maps_get_unsqueezed(self):
        scores = [0.5]
        maps = torch.randn(1, 64, 64)  # 2D per sample (no channel dim)
        results = build_predict_results(None, scores, maps)
        assert results[0].pred_anomaly_map.shape == (1, 64, 64)

    def test_with_existing_data_samples(self):
        samples = [ADDataSample(), ADDataSample()]
        samples[0].gt_label = 0
        samples[1].gt_label = 1
        scores = [0.2, 0.8]
        results = build_predict_results(samples, scores)
        assert results[0].gt_label == 0
        assert results[0].pred_score == pytest.approx(0.2, abs=1e-5)
        assert results[1].gt_label == 1

    def test_no_score_maps(self):
        scores = [0.3, 0.7]
        results = build_predict_results(None, scores, None)
        assert len(results) == 2
        assert results[0].pred_score == pytest.approx(0.3, abs=1e-5)
        assert not hasattr(results[0], 'pred_anomaly_map')

    def test_score_maps_on_cpu(self):
        scores = torch.tensor([0.5])
        maps = torch.randn(1, 1, 32, 32)
        results = build_predict_results(None, scores, maps)
        assert results[0].pred_anomaly_map.device == torch.device('cpu')

    def test_single_sample(self):
        scores = [1.0]
        maps = torch.randn(1, 1, 16, 16)
        results = build_predict_results(None, scores, maps)
        assert len(results) == 1
        assert results[0].pred_score == 1.0

    def test_extra_scores_are_attached(self):
        scores = [0.8]
        maps = torch.randn(1, 1, 16, 16)
        results = build_predict_results(
            None,
            scores,
            maps,
            extra_scores={'pred_score_mean': [0.5], 'pred_score_max': [0.8]},
        )
        assert results[0].pred_score == pytest.approx(0.8, abs=1e-5)
        assert results[0].pred_score_mean == pytest.approx(0.5, abs=1e-5)
        assert results[0].pred_score_max == pytest.approx(0.8, abs=1e-5)

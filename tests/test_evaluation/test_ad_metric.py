"""Tests for AnomalyDetectionMetric."""

import numpy as np
import pytest
import torch
from baoiad.structures import ADDataSample

import baoiad  # noqa: F401

from baoiad.evaluation.ad_metric import AnomalyDetectionMetric
from baoiad.evaluation.anomaly_map_mean_metric import AnomalyMapMeanMetric


def _make_sample(pred_score, gt_label, cls_name='bottle', H=32, W=32, anomalous_region=False):
    s = ADDataSample()
    s.pred_score = pred_score
    s.gt_label = gt_label
    s.cls_name = cls_name
    s.defect_type = 'good' if gt_label == 0 else 'broken'
    if anomalous_region and gt_label == 1:
        mask = np.zeros((H, W), dtype=np.float32)
        mask[8:24, 8:24] = 1.0
        amap = np.random.rand(H, W).astype(np.float32)
        amap[8:24, 8:24] += 1.0
    else:
        mask = np.zeros((H, W), dtype=np.float32)
        amap = np.full((H, W), pred_score, dtype=np.float32)
    s.gt_mask = torch.from_numpy(mask)
    s.pred_anomaly_map = torch.from_numpy(amap)
    return s


class TestAnomalyDetectionMetric:
    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError, match='Unsupported metrics'):
            AnomalyDetectionMetric(metrics=['image_auroc', 'typo_metric'])

    def test_process_and_compute(self):
        metric = AnomalyDetectionMetric(metrics=['image_auroc'])
        samples = [_make_sample(0.1, 0), _make_sample(0.9, 1)]
        metric.process({}, samples)
        result = metric.compute_metrics(metric.results)
        assert 'image_auroc' in result
        assert 0.0 <= result['image_auroc'] <= 1.0

    def test_perfect_predictions(self):
        metric = AnomalyDetectionMetric(metrics=['image_auroc'])
        samples = ([_make_sample(0.0, 0) for _ in range(5)] +
                   [_make_sample(1.0, 1) for _ in range(5)])
        metric.process({}, samples)
        result = metric.compute_metrics(metric.results)
        assert result['image_auroc'] == 1.0

    def test_random_predictions(self):
        metric = AnomalyDetectionMetric(metrics=['image_auroc'])
        rng = np.random.default_rng(0)
        samples = [_make_sample(float(rng.random()), i % 2) for i in range(20)]
        metric.process({}, samples)
        result = metric.compute_metrics(metric.results)
        assert 0.0 <= result['image_auroc'] <= 1.0

    def test_single_class_all_normal(self):
        metric = AnomalyDetectionMetric(metrics=['image_auroc'])
        samples = [_make_sample(0.1, 0) for _ in range(5)]
        metric.process({}, samples)
        result = metric.compute_metrics(metric.results)
        # Only one class → safe fallback to 0.0
        assert result['image_auroc'] == 0.0

    def test_non_square_masks(self):
        metric = AnomalyDetectionMetric(metrics=['pixel_auroc'])
        for gt in [0, 1]:
            s = _make_sample(0.5, gt, H=32, W=48, anomalous_region=(gt == 1))
            metric.process({}, [s])
        # Should not raise
        metric.compute_metrics(metric.results)

    def test_multi_class_results(self):
        """Per-class results should appear in output."""
        metric = AnomalyDetectionMetric(metrics=['image_auroc'])
        samples = ([_make_sample(0.0, 0, cls_name='bottle')] * 3 +
                   [_make_sample(1.0, 1, cls_name='bottle')] * 3 +
                   [_make_sample(0.0, 0, cls_name='cable')] * 3 +
                   [_make_sample(1.0, 1, cls_name='cable')] * 3)
        metric.process({}, samples)
        result = metric.compute_metrics(metric.results)
        assert 'bottle/image_auroc' in result
        assert 'cable/image_auroc' in result
        assert result['bottle/image_auroc'] == 1.0
        assert result['cable/image_auroc'] == 1.0

    def test_macro_average_matches_per_class_mean(self):
        metric = AnomalyDetectionMetric(metrics=['image_auroc'])
        samples = [
            _make_sample(0.0, 0, cls_name='bottle'),
            _make_sample(1.0, 1, cls_name='bottle'),
            _make_sample(1.0, 0, cls_name='cable'),
            _make_sample(0.0, 1, cls_name='cable'),
        ]
        metric.process({}, samples)
        result = metric.compute_metrics(metric.results)
        assert result['bottle/image_auroc'] == 1.0
        assert result['cable/image_auroc'] == 0.0
        assert result['image_auroc'] == pytest.approx(0.5)

    @pytest.mark.parametrize('metric_name', [
        'image_auroc', 'pixel_auroc', 'image_f1max', 'pixel_f1max',
        'image_ap', 'pixel_ap', 'image_ece', 'pixel_ece', 'image_fpr@95tpr',
    ])
    def test_each_metric_type(self, metric_name):
        """Each metric type should compute without error and return [0, 1]."""
        metric = AnomalyDetectionMetric(metrics=[metric_name])
        samples = ([_make_sample(0.1, 0, anomalous_region=False)] * 5 +
                   [_make_sample(0.9, 1, anomalous_region=True)] * 5)
        metric.process({}, samples)
        result = metric.compute_metrics(metric.results)
        assert metric_name in result
        assert 0.0 <= result[metric_name] <= 1.0, f'{metric_name}={result[metric_name]}'

    def test_pixel_level_perfect(self):
        """Perfect pixel-level predictions → pixel_auroc = 1.0."""
        metric = AnomalyDetectionMetric(metrics=['pixel_auroc'])
        for i in range(10):
            gt = i % 2
            s = ADDataSample()
            s.pred_score = float(gt)
            s.gt_label = gt
            s.cls_name = 'bottle'
            mask = np.zeros((32, 32), dtype=np.float32)
            amap = np.zeros((32, 32), dtype=np.float32)
            if gt == 1:
                mask[8:24, 8:24] = 1.0
                amap[8:24, 8:24] = 1.0
            s.gt_mask = torch.from_numpy(mask)
            s.pred_anomaly_map = torch.from_numpy(amap)
            metric.process({}, [s])
        result = metric.compute_metrics(metric.results)
        assert result['pixel_auroc'] == 1.0

    def test_aupro_through_metric(self):
        """AUPRO should work through the unified metric interface."""
        metric = AnomalyDetectionMetric(metrics=['aupro'])
        samples = ([_make_sample(0.1, 0)] * 3 +
                   [_make_sample(0.9, 1, anomalous_region=True)] * 3)
        metric.process({}, samples)
        result = metric.compute_metrics(metric.results)
        assert 'aupro' in result
        assert 0.0 <= result['aupro'] <= 1.0

    def test_anomaly_map_mean_metric_matches_global_map_mean(self):
        metric = AnomalyMapMeanMetric()

        sample_a = ADDataSample()
        sample_a.pred_anomaly_map = torch.ones(1, 2, 2)

        sample_b = ADDataSample()
        sample_b.pred_anomaly_map = torch.full((1, 2, 2), 3.0)

        metric.process({}, [sample_a, sample_b])
        result = metric.compute_metrics(metric.results)
        assert result['score_mean'] == pytest.approx(2.0)

    def test_anomaly_map_mean_metric_prefers_raw_map_when_available(self):
        metric = AnomalyMapMeanMetric()

        sample = ADDataSample()
        sample.pred_anomaly_map = torch.full((1, 4, 4), 100.0)
        sample.pred_anomaly_map_raw = torch.full((1, 2, 2), 3.0)

        metric.process({}, [sample])
        result = metric.compute_metrics(metric.results)
        assert result['score_mean'] == pytest.approx(3.0)

    def test_resize_mask_resizes_predictions_and_gt(self):
        metric = AnomalyDetectionMetric(metrics=['pixel_auroc'], resize_mask=16)
        sample = _make_sample(0.9, 1, H=32, W=32, anomalous_region=True)
        metric.process({}, [sample])
        assert metric.results[0]['gt_mask_shape'] == (16, 16)
        assert metric.results[0]['gt_mask'].shape[0] == 16 * 16
        assert metric.results[0]['pred_anomaly_map'].shape[0] == 16 * 16

    def test_resize_mask_can_use_bilinear_gt_and_threshold(self):
        metric = AnomalyDetectionMetric(
            metrics=['pixel_auroc'],
            resize_mask=16,
            resize_gt_mask_mode='bilinear',
            resize_gt_mask_threshold=0.5,
        )
        sample = _make_sample(0.9, 1, H=8, W=8, anomalous_region=True)
        metric.process({}, [sample])
        gt = metric.results[0]['gt_mask']
        assert gt.shape[0] == 16 * 16
        assert set(np.unique(gt).tolist()).issubset({0.0, 1.0})

    def test_resize_gt_mask_mode_must_be_supported(self):
        with pytest.raises(ValueError, match='resize_gt_mask_mode'):
            AnomalyDetectionMetric(metrics=['pixel_auroc'], resize_gt_mask_mode='bad')

    def test_pixel_metrics_accept_fractional_gt_masks(self):
        metric = AnomalyDetectionMetric(metrics=['pixel_auroc', 'pixel_ap'])

        good = ADDataSample()
        good.pred_score = 0.1
        good.gt_label = 0
        good.cls_name = 'bottle'
        good.gt_mask = torch.zeros(8, 8)
        good.pred_anomaly_map = torch.zeros(8, 8)

        bad = ADDataSample()
        bad.pred_score = 0.9
        bad.gt_label = 1
        bad.cls_name = 'bottle'
        frac_mask = torch.zeros(8, 8)
        frac_mask[2:6, 2:6] = 1.0
        frac_mask[1, 2:6] = 0.4
        frac_mask[6, 2:6] = 0.4
        frac_mask[2:6, 1] = 0.4
        frac_mask[2:6, 6] = 0.4
        bad.gt_mask = frac_mask
        amap = torch.zeros(8, 8)
        amap[2:6, 2:6] = 1.0
        bad.pred_anomaly_map = amap

        metric.process({}, [good, bad])
        result = metric.compute_metrics(metric.results)
        assert result['pixel_auroc'] > 0.99
        assert result['pixel_ap'] > 0.99

    def test_normalize_image_scores_applies_per_class_minmax(self):
        metric = AnomalyDetectionMetric(metrics=['image_ap'], normalize_image_scores=True)
        samples = [
            _make_sample(10.0, 0, cls_name='bottle'),
            _make_sample(30.0, 1, cls_name='bottle'),
            _make_sample(5.0, 1, cls_name='cable'),
            _make_sample(5.0, 0, cls_name='cable'),
        ]
        metric.process({}, samples)
        result = metric.compute_metrics(metric.results)
        assert result['bottle/image_ap'] == 1.0
        assert result['cable/image_ap'] == pytest.approx(0.5)
        assert result['image_ap'] == pytest.approx(0.75)

    def test_normalize_image_scores_handles_constant_scores(self):
        metric = AnomalyDetectionMetric(metrics=['image_auroc'], normalize_image_scores=True)
        samples = [_make_sample(5.0, 0), _make_sample(5.0, 1)]
        metric.process({}, samples)
        result = metric.compute_metrics(metric.results)
        assert result['image_auroc'] == pytest.approx(0.5)
        assert metric._minmax_normalize(np.array([5.0, 5.0])).tolist() == [0.0, 0.0]

    def test_image_score_field_can_switch_primary_image_metrics(self):
        metric = AnomalyDetectionMetric(
            metrics=['image_auroc', 'image_ap', 'image_auroc_mean', 'image_auroc_max'],
            image_score_field='pred_score_max',
        )
        good = _make_sample(0.9, 0)
        bad = _make_sample(0.1, 1)
        good.pred_score_max = 0.1
        bad.pred_score_max = 0.9
        good.pred_score_mean = 0.2
        bad.pred_score_mean = 0.8

        metric.process({}, [good, bad])
        result = metric.compute_metrics(metric.results)
        assert result['image_auroc'] == pytest.approx(1.0)
        assert result['image_ap'] == pytest.approx(1.0)
        assert result['image_auroc_mean'] == pytest.approx(1.0)
        assert result['image_auroc_max'] == pytest.approx(1.0)

    def test_normalize_pred_maps_batch_broadcast_matches_reference_formula(self):
        metric = AnomalyDetectionMetric(metrics=['pixel_auroc'], normalize_pred_maps='batch_broadcast')
        pred_maps = [
            np.array([[1.0, 3.0], [1.0, 3.0]]),
            np.array([[10.0, 14.0], [10.0, 14.0]]),
        ]

        normalized = metric._normalize_pred_maps_2d(pred_maps, 'batch_broadcast')
        expected = (
            (np.array(pred_maps) - 1.0) / 2.0 +
            (np.array(pred_maps) - 10.0) / 4.0
        ) / 2.0

        assert np.allclose(normalized, expected)

    def test_pixel_metrics_support_mixed_spatial_shapes_within_one_class(self):
        metric = AnomalyDetectionMetric(
            metrics=['pixel_auroc', 'pixel_ap', 'pixel_ece', 'aupro', 'aupimo']
        )

        good = ADDataSample()
        good.pred_score = 0.1
        good.gt_label = 0
        good.cls_name = 'bottle'
        good.gt_mask = torch.zeros(8, 8)
        good.pred_anomaly_map = torch.full((8, 8), 0.1)

        bad = ADDataSample()
        bad.pred_score = 0.9
        bad.gt_label = 1
        bad.cls_name = 'bottle'
        bad.gt_mask = torch.zeros(6, 10)
        bad.gt_mask[1:5, 3:7] = 1.0
        bad.pred_anomaly_map = torch.zeros(6, 10)
        bad.pred_anomaly_map[1:5, 3:7] = 0.9

        metric.process({}, [good, bad])
        result = metric.compute_metrics(metric.results)

        assert result['pixel_auroc'] > 0.99
        assert result['pixel_ap'] > 0.99
        assert 0.0 <= result['pixel_ece'] <= 1.0
        assert 0.0 <= result['aupro'] <= 1.0
        assert 0.0 <= result['aupimo'] <= 1.0

    def test_flip_auroc_if_below_half_matches_official_behavior(self):
        good = ADDataSample()
        good.pred_score = 1.0
        good.gt_label = 0
        good.cls_name = 'bottle'
        good.gt_mask = torch.zeros(8, 8)
        good.pred_anomaly_map = torch.ones(8, 8)

        bad = ADDataSample()
        bad.pred_score = 0.0
        bad.gt_label = 1
        bad.cls_name = 'bottle'
        bad.gt_mask = torch.zeros(8, 8)
        bad.gt_mask[2:6, 2:6] = 1.0
        bad.pred_anomaly_map = torch.ones(8, 8)
        bad.pred_anomaly_map[2:6, 2:6] = 0.0

        raw_metric = AnomalyDetectionMetric(metrics=['image_auroc', 'pixel_auroc'])
        raw_metric.process({}, [good, bad])
        raw_result = raw_metric.compute_metrics(raw_metric.results)
        assert raw_result['image_auroc'] == pytest.approx(0.0)
        assert raw_result['pixel_auroc'] == pytest.approx(0.0)

        flipped_metric = AnomalyDetectionMetric(
            metrics=['image_auroc', 'pixel_auroc'],
            flip_auroc_if_below_half=True,
        )
        flipped_metric.process({}, [good, bad])
        flipped_result = flipped_metric.compute_metrics(flipped_metric.results)
        assert flipped_result['image_auroc'] == pytest.approx(1.0)
        assert flipped_result['pixel_auroc'] == pytest.approx(1.0)

    def test_metric_supports_mean_and_max_image_scores(self):
        metric = AnomalyDetectionMetric(metrics=['image_auroc', 'image_auroc_mean', 'image_auroc_max'])

        good = _make_sample(0.1, 0)
        good.pred_score_mean = 0.2
        good.pred_score_max = 0.1

        bad = _make_sample(0.9, 1)
        bad.pred_score_mean = 0.8
        bad.pred_score_max = 0.9

        metric.process({}, [good, bad])
        result = metric.compute_metrics(metric.results)
        assert result['image_auroc'] == pytest.approx(1.0)
        assert result['image_auroc_mean'] == pytest.approx(1.0)
        assert result['image_auroc_max'] == pytest.approx(1.0)

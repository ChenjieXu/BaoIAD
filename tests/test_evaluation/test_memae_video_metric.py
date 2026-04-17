"""Tests for MemAE video metric."""

import pytest

import baoiad  # noqa: F401
from baoiad.registry import METRICS
from baoiad.structures import ADDataSample


def test_memae_video_metric_matches_official_per_video_normalization():
    metric = METRICS.build(dict(type='MemAEVideoMetric'))

    samples = []
    for video_name in ['Test001', 'Test002']:
        normal = ADDataSample()
        normal.set_metainfo(dict(cls_name='UCSDped2', video_name=video_name, frame_idx=9))
        normal.gt_label = 0
        normal.pred_score = 0.1
        samples.append(normal)

        anomaly = ADDataSample()
        anomaly.set_metainfo(dict(cls_name='UCSDped2', video_name=video_name, frame_idx=10))
        anomaly.gt_label = 1
        anomaly.pred_score = 0.9
        samples.append(anomaly)

    metric.process(data_batch={}, data_samples=samples)
    results = metric.compute_metrics(metric.results)
    assert results['image_auroc'] == pytest.approx(1.0)


def test_memae_video_metric_accepts_dict_samples():
    metric = METRICS.build(dict(type='MemAEVideoMetric'))
    samples = [
        dict(pred_score=0.1, gt_label=0, cls_name='UCSDped2', video_name='Test001', frame_idx=9),
        dict(pred_score=0.9, gt_label=1, cls_name='UCSDped2', video_name='Test001', frame_idx=10),
    ]
    metric.process(data_batch={}, data_samples=samples)
    results = metric.compute_metrics(metric.results)
    assert 'image_auroc' in results

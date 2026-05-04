"""Tests for ADVisualizer."""

import numpy as np

import baoiad  # noqa: F401

from baoiad.visualization.ad_visualizer import ADVisualizer


class TestADVisualizer:
    def test_draw_anomaly_map(self):
        vis = ADVisualizer(name='test_vis')
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        amap = np.random.rand(64, 64).astype(np.float32)
        gt_mask = np.zeros((64, 64), dtype=np.float32)
        gt_mask[16:48, 16:48] = 1.0

        out = vis.draw_anomaly_map(image, amap, gt_mask=gt_mask)
        assert out.shape == (64, 64, 3)
        assert out.dtype == np.uint8

    def test_draw_without_gt_mask(self):
        vis = ADVisualizer(name='test_vis2')
        image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        amap = np.random.rand(32, 32).astype(np.float32)
        out = vis.draw_anomaly_map(image, amap)
        assert out.shape == (32, 32, 3)

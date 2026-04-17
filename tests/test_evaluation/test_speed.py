"""Tests for speed measurement."""

import torch
import torch.nn as nn

from baoiad.evaluation.speed import measure_speed


class TestMeasureSpeed:
    def test_measure_speed(self):
        model = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256))
        result = measure_speed(model, input_size=(8, 256), device='cpu', n_warmup=2, n_runs=10)
        assert 'fps' in result
        assert result['fps'] > 0
        assert result['latency_ms'] >= 0

"""Validation metric for official STFPM-style score-map selection."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from mmengine.evaluator import BaseMetric

from baoiad.registry import METRICS


@METRICS.register_module(force=True)
class AnomalyMapMeanMetric(BaseMetric):
    """Compute the mean anomaly-map value across a validation set.

    This matches the STFPM official repository, which selects the best
    checkpoint by minimizing ``score_map.mean()`` on a held-out subset of the
    clean training images.
    """

    default_prefix = 'ad'

    def process(self, data_batch: dict, data_samples: Sequence) -> None:
        del data_batch
        for sample in data_samples:
            if isinstance(sample, dict):
                pred_map = sample.get('pred_anomaly_map_raw', sample['pred_anomaly_map'])
            else:
                pred_map = getattr(
                    sample,
                    'pred_anomaly_map_raw',
                    getattr(sample, 'pred_anomaly_map'),
                )
            if isinstance(pred_map, np.ndarray):
                pred_map = torch.from_numpy(pred_map)
            if not torch.is_tensor(pred_map):
                pred_map = torch.as_tensor(pred_map)
            pred_map = pred_map.detach().cpu().float()
            self.results.append({
                'sum': float(pred_map.sum().item()),
                'numel': int(pred_map.numel()),
            })

    def compute_metrics(self, results: list[dict]) -> dict:
        total_sum = sum(float(item['sum']) for item in results)
        total_numel = sum(int(item['numel']) for item in results)
        score_mean = 0.0 if total_numel == 0 else total_sum / total_numel
        return {'score_mean': float(score_mean)}

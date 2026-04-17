"""Official-style MemAE video metric."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric
from sklearn.metrics import roc_auc_score

from baoiad.registry import METRICS


@METRICS.register_module()
class MemAEVideoMetric(BaseMetric):
    """Frame-level AUROC following MemAE's official video evaluation."""

    default_prefix = 'ad'

    def process(self, data_batch: dict, data_samples: Sequence) -> None:
        for sample in data_samples:
            is_dict = isinstance(sample, dict)

            def _get(key: str, default=None):
                if is_dict:
                    return sample.get(key, default)
                return getattr(sample, key, default)

            self.results.append(
                dict(
                    pred_score=float(_get('pred_score')),
                    gt_label=int(_get('gt_label')),
                    cls_name=str(_get('cls_name', '')),
                    video_name=str(_get('video_name', '')),
                    frame_idx=int(_get('frame_idx', 0)),
                )
            )

    @staticmethod
    def _official_video_regularities(scores: np.ndarray) -> np.ndarray:
        score_min = float(scores.min())
        score_max = float(scores.max())
        if score_max <= score_min:
            return np.ones_like(scores, dtype=np.float64)
        return 1.0 - ((scores - score_min) / (score_max - score_min))

    def _compute_single_class(self, samples: List[Dict]) -> Dict[str, float]:
        per_video: dict[str, list[Dict]] = defaultdict(list)
        for sample in samples:
            per_video[sample['video_name']].append(sample)

        all_regularities = []
        all_normal_labels = []
        for records in per_video.values():
            ordered = sorted(records, key=lambda item: item['frame_idx'])
            scores = np.asarray([item['pred_score'] for item in ordered], dtype=np.float64)
            labels = np.asarray([item['gt_label'] for item in ordered], dtype=np.int64)
            regularities = self._official_video_regularities(scores)
            normal_labels = (labels == 0).astype(np.int64)
            all_regularities.extend(regularities.tolist())
            all_normal_labels.extend(normal_labels.tolist())

        y_true = np.asarray(all_normal_labels, dtype=np.int64)
        y_score = np.asarray(all_regularities, dtype=np.float64)
        if len(np.unique(y_true)) < 2:
            auc = 0.0
        else:
            auc = float(roc_auc_score(y_true, y_score))
        return dict(image_auroc=auc, num_videos=float(len(per_video)))

    def compute_metrics(self, results: List[Dict]) -> Dict:
        grouped: Dict[str, List[Dict]] = defaultdict(list)
        for result in results:
            grouped[result['cls_name']].append(result)

        out: Dict[str, float] = {}
        per_class = {cls_name: self._compute_single_class(samples) for cls_name, samples in grouped.items()}
        image_aurocs = [metrics['image_auroc'] for metrics in per_class.values()]
        out['image_auroc'] = float(np.mean(image_aurocs)) if image_aurocs else 0.0
        for cls_name, metrics in per_class.items():
            for key, value in metrics.items():
                out[f'{cls_name}/{key}'] = value
        return out

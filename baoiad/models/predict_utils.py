"""Utilities for building prediction results."""

from typing import Mapping, Optional, Sequence

import numpy as np
import torch

from baoiad.structures import ADDataSample


def _normalize_scores(img_scores) -> torch.Tensor:
    if isinstance(img_scores, torch.Tensor):
        return img_scores.reshape(-1).detach().cpu()
    if isinstance(img_scores, np.ndarray):
        return torch.from_numpy(img_scores.reshape(-1))
    return torch.tensor(list(img_scores), dtype=torch.float32)


def _extract_score_map(score_maps, index: int) -> Optional[torch.Tensor]:
    if score_maps is None:
        return None

    if isinstance(score_maps, torch.Tensor):
        if score_maps.ndim == 2:
            score_map = score_maps
        elif score_maps.ndim == 3:
            score_map = score_maps[index]
        elif score_maps.ndim == 4:
            score_map = score_maps[index]
        else:
            raise ValueError(f'Unsupported score_maps tensor shape: {tuple(score_maps.shape)}')
    else:
        score_map = score_maps[index]

    if isinstance(score_map, np.ndarray):
        score_map = torch.from_numpy(score_map)
    if not isinstance(score_map, torch.Tensor):
        score_map = torch.tensor(score_map)

    score_map = score_map.detach().cpu()
    if score_map.ndim == 2:
        score_map = score_map.unsqueeze(0)
    elif score_map.ndim == 3 and score_map.shape[0] != 1:
        # Keep [C, H, W] format if multi-channel maps are used.
        pass
    elif score_map.ndim != 3:
        raise ValueError(f'Unsupported per-sample anomaly map shape: {tuple(score_map.shape)}')
    return score_map


def build_predict_results(
    data_samples: Optional[Sequence[ADDataSample]],
    img_scores,
    score_maps=None,
    extra_scores: Optional[Mapping[str, object]] = None,
):
    """Build :class:`ADDataSample` predictions from scores and maps."""
    scores = _normalize_scores(img_scores)
    num_samples = int(scores.shape[0])
    results = []
    normalized_extra_scores = {
        key: _normalize_scores(value)
        for key, value in (extra_scores or {}).items()
    }

    for i in range(num_samples):
        if data_samples is not None and i < len(data_samples) and data_samples[i] is not None:
            sample = data_samples[i]
        else:
            sample = ADDataSample()

        sample.pred_score = float(scores[i].item())
        for key, value in normalized_extra_scores.items():
            if value.shape[0] != num_samples:
                raise ValueError(
                    f'Extra score field {key!r} has length {value.shape[0]}, '
                    f'expected {num_samples}.')
            setattr(sample, key, float(value[i].item()))
        score_map = _extract_score_map(score_maps, i)
        if score_map is not None:
            sample.pred_anomaly_map = score_map
        results.append(sample)

    return results

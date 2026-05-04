"""Strict RegAD evaluation helpers."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

_RESAMPLING = getattr(Image, 'Resampling', Image)
_RGB_RESAMPLE = _RESAMPLING.LANCZOS
SUPPORT_SET_ENV_VAR = 'REGAD_SUPPORT_SET_ROOT'


def _load_rgb_tensor(img_path: str, img_size: int) -> torch.Tensor:
    img = Image.open(img_path).convert('RGB')
    img = img.resize((img_size, img_size), _RGB_RESAMPLE)
    array = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1)).contiguous()


def _normalize_support_round_tensor(round_tensor: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(round_tensor):
        round_tensor = torch.as_tensor(round_tensor)
    round_tensor = round_tensor.detach().cpu().float()
    if round_tensor.dim() == 5 and round_tensor.size(0) == 1:
        round_tensor = round_tensor.squeeze(0)
    if round_tensor.dim() != 4:
        raise ValueError(f'Expected support round tensor with shape (K,C,H,W), got {tuple(round_tensor.shape)}')
    if float(round_tensor.max().item()) > 1.5:
        round_tensor = round_tensor / 255.0
    return round_tensor


def _missing_support_set_error(support_file: Path) -> FileNotFoundError:
    return FileNotFoundError(
        'RegAD strict requires the official support-set rounds, but the expected file was '
        f'not found: {support_file}. Set {SUPPORT_SET_ENV_VAR} to the extracted '
        'support_set directory or run `python tools/fetch_regad_support_set.py`.'
    )


def _corrupt_support_set_error(support_file: Path, reason: str) -> RuntimeError:
    return RuntimeError(
        'RegAD strict found an official support-set file, but it is unreadable or empty: '
        f'{support_file} ({reason}). Verify the downloaded archive and audit it with '
        '`runs/alignment/regad_support_set_audit.json` or re-extract a clean support_set.'
    )


def load_or_sample_support_rounds(
    *,
    data_root: str,
    target_cls: str,
    img_size: int,
    shot: int,
    inferences: int,
    seed: int,
    support_set_root: str | None = None,
    allow_fallback: bool = True,
) -> tuple[List[torch.Tensor], str, Path | None]:
    """Load official fixed support rounds or deterministically sample a fallback."""
    support_file: Path | None = None
    if support_set_root:
        support_file = Path(os.path.expanduser(support_set_root)) / target_cls / f'{shot}_{inferences}.pt'
        if support_file.is_file():
            if support_file.stat().st_size == 0:
                raise _corrupt_support_set_error(support_file, 'file size is 0 bytes')
            try:
                raw_rounds = torch.load(str(support_file), map_location='cpu')
            except EOFError as exc:
                raise _corrupt_support_set_error(support_file, 'EOF while reading torch serialization') from exc
            rounds = [_normalize_support_round_tensor(round_tensor) for round_tensor in raw_rounds]
            if not rounds:
                raise _corrupt_support_set_error(support_file, 'no support rounds stored in file')
            return rounds, 'official', support_file
        if not allow_fallback:
            raise _missing_support_set_error(support_file)
    elif not allow_fallback:
        raise _missing_support_set_error(
            Path('data') / 'regad_official' / 'support_set' / target_cls / f'{shot}_{inferences}.pt'
        )

    train_dir = Path(data_root) / target_cls / 'train' / 'good'
    if not train_dir.is_dir():
        raise FileNotFoundError(f'RegAD support directory not found: {train_dir}')

    img_paths = sorted(
        str(path) for path in train_dir.iterdir()
        if path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    )
    if not img_paths:
        raise FileNotFoundError(f'No support images found under {train_dir}')

    rng = random.Random(seed)
    rounds = []
    for _ in range(inferences):
        sampled_paths = [rng.choice(img_paths) for _ in range(shot)]
        rounds.append(torch.stack([
            _load_rgb_tensor(img_path, img_size) for img_path in sampled_paths
        ], dim=0))
    return rounds, 'fallback', support_file


def compute_regad_metrics(
    *,
    score_maps: np.ndarray,
    gt_labels: Iterable[int],
    gt_masks: np.ndarray,
) -> dict[str, float]:
    """Apply official per-round min-max normalization and compute AUROC metrics."""
    if score_maps.ndim != 3:
        raise ValueError(f'score_maps must have shape (N,H,W), got {score_maps.shape}')

    max_score = float(score_maps.max())
    min_score = float(score_maps.min())
    denom = max(max_score - min_score, 1e-12)
    normalized = (score_maps - min_score) / denom

    img_scores = normalized.reshape(normalized.shape[0], -1).max(axis=1)
    gt_labels = np.asarray(list(gt_labels), dtype=np.int64)
    gt_masks = np.asarray(gt_masks)
    gt_masks = (gt_masks > 0.5).astype(np.int64)

    return {
        'image_auroc': float(roc_auc_score(gt_labels, img_scores)),
        'pixel_auroc': float(roc_auc_score(gt_masks.flatten(), normalized.flatten())),
    }

"""Shared UniVAD asset and mask helpers.

These helpers centralize the on-disk layout for precomputed UniVAD assets so
the detector and preprocessing script do not drift apart.
"""

from __future__ import annotations

import os
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch


def relative_data_path(img_path: str) -> str:
    """Return the path relative to the dataset root's ``data/`` prefix."""
    normalized = os.path.normpath(img_path).replace('\\', '/')
    if '/data/' in normalized:
        return normalized.split('/data/', 1)[-1]
    if normalized.startswith('data/'):
        return normalized[len('data/'):]
    return normalized


def decompose_image_path(img_path: str) -> Tuple[str, str, str, str]:
    """Split an anomaly image path into class / split / defect / stem."""
    normalized = os.path.normpath(img_path)
    parts = normalized.split(os.sep)
    if len(parts) < 4:
        raise ValueError(f'Unsupported UniVAD image path: {img_path}')
    stem = os.path.splitext(parts[-1])[0]
    defect_type = parts[-2]
    split = parts[-3]
    cls_name = parts[-4]
    return cls_name, split, defect_type, stem


def mask_path_candidates(mask_dir: str, cls_name: str, img_path: str) -> List[str]:
    """Return candidate paths for a C3 grounding/component mask."""
    _cls_from_path, split, defect_type, stem = decompose_image_path(img_path)
    rel_stem = os.path.splitext(relative_data_path(img_path))[0]
    candidates = [
        os.path.join(mask_dir, cls_name, split, defect_type, f'{stem}.npy'),
        os.path.join(mask_dir, cls_name, split, defect_type, stem, 'grounding_mask.png'),
        os.path.join(mask_dir, cls_name, split, defect_type, f'{stem}.png'),
        os.path.join(mask_dir, rel_stem, 'grounding_mask.png'),
        os.path.join(mask_dir, f'{rel_stem}.npy'),
    ]
    return _dedupe_paths(candidates)


def heat_mask_path_candidates(heat_mask_dir: str, cls_name: str, img_path: str) -> List[str]:
    """Return candidate paths for a refined MULTI heat-mask asset."""
    _cls_from_path, split, defect_type, stem = decompose_image_path(img_path)
    rel_stem = os.path.splitext(relative_data_path(img_path))[0]
    candidates = [
        os.path.join(heat_mask_dir, cls_name, split, defect_type, f'{stem}.png'),
        os.path.join(heat_mask_dir, cls_name, split, defect_type, f'{stem}.npy'),
        os.path.join(heat_mask_dir, f'{rel_stem}.png'),
        os.path.join(heat_mask_dir, f'{rel_stem}.npy'),
    ]
    return _dedupe_paths(candidates)


def train_features_path_candidates(heat_mask_dir: str, cls_name: str) -> List[str]:
    """Return candidate paths for saved MULTI train prototypes."""
    candidates = [
        os.path.join(heat_mask_dir, cls_name, 'train_features_sampled.pth'),
        os.path.join(heat_mask_dir, f'{cls_name}_heat', 'train_features_sampled.pth'),
    ]
    return _dedupe_paths(candidates)


def compose_labeled_mask(masks: Sequence[np.ndarray]) -> np.ndarray:
    """Compose a list of binary masks into a 1-indexed label map."""
    if not masks:
        raise ValueError('Expected at least one mask to compose.')
    result = np.zeros_like(masks[0], dtype=np.int32)
    for idx, mask in enumerate(masks, start=1):
        result[np.asarray(mask) > 0] = idx
    return result


def split_masks_from_one_mask(mask: np.ndarray, *, min_ratio: float = 1e-4) -> Tuple[List[np.ndarray], List[int]]:
    """Split a label map into binary component masks, excluding background."""
    return _split_masks(mask, include_background=False, min_ratio=min_ratio)


def split_masks_from_one_mask_with_bg(
    mask: np.ndarray,
    *,
    min_ratio: float = 1e-4,
) -> Tuple[List[np.ndarray], List[int]]:
    """Split a label map into binary masks, keeping background as label 0."""
    return _split_masks(mask, include_background=True, min_ratio=min_ratio)


def split_masks_from_one_mask_torch(mask: torch.Tensor, *, min_ratio: float = 1e-3) -> List[torch.Tensor]:
    """Torch variant used by UniVAD's gate and heat-mask helpers."""
    if mask.ndim != 2:
        raise ValueError(f'Expected 2D mask tensor, got shape {tuple(mask.shape)}')
    height, width = mask.shape
    result = []
    max_label = int(mask.max().item()) if mask.numel() > 0 else 0
    for label in range(1, max_label + 1):
        component = torch.zeros_like(mask)
        component[mask == label] = 255
        if torch.sum(component != 0).item() / float(height * width) > min_ratio:
            result.append(component)
    return result


def assign_fine_to_coarse_torch(coarse_masks: torch.Tensor, fine_masks: torch.Tensor) -> torch.Tensor:
    """Assign fine masks to the closest coarse mask, matching UniVAD reference."""
    num_coarse, height, width = coarse_masks.shape
    num_fine = fine_masks.shape[0]
    coarse_to_fine = {idx: [] for idx in range(num_coarse)}

    for fine_idx in range(num_fine):
        if num_fine > 1:
            if fine_masks[fine_idx][0, 0] and fine_masks[fine_idx][height - 1, width - 1]:
                continue
            probe = min(10, max(height - 1, 0), max(width - 1, 0))
            tail_h = max(height - probe - 1, 0)
            tail_w = max(width - probe - 1, 0)
            if fine_masks[fine_idx][probe, probe] and fine_masks[fine_idx][tail_h, tail_w]:
                continue
        best_overlap = 0.0
        best_idx = -1
        for coarse_idx in range(num_coarse):
            overlap = torch.sum((fine_masks[fine_idx] & coarse_masks[coarse_idx]).float()).item()
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = coarse_idx
        if best_idx != -1:
            coarse_to_fine[best_idx].append(fine_masks[fine_idx])

    new_masks = torch.zeros_like(coarse_masks)
    for coarse_idx, masks in coarse_to_fine.items():
        for fine_mask in masks:
            new_masks[coarse_idx][fine_mask > 0] = coarse_idx + 1
    return new_masks


def _split_masks(mask: np.ndarray, *, include_background: bool, min_ratio: float) -> Tuple[List[np.ndarray], List[int]]:
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f'Expected 2D mask array, got shape {mask.shape}')
    start = 0 if include_background else 1
    max_label = int(mask.max()) if mask.size > 0 else 0
    result_masks: List[np.ndarray] = []
    result_idxs: List[int] = []
    for label in range(start, max_label + 1):
        component = np.zeros_like(mask, dtype=np.uint8)
        component[mask == label] = 255
        if np.sum(component != 0) / float(component.size) > min_ratio:
            result_masks.append(component)
            result_idxs.append(label)
    return result_masks, result_idxs


def _dedupe_paths(paths: Iterable[str]) -> List[str]:
    deduped: List[str] = []
    seen = set()
    for path in paths:
        norm_path = os.path.normpath(path)
        if norm_path not in seen:
            deduped.append(norm_path)
            seen.add(norm_path)
    return deduped


__all__ = [
    'assign_fine_to_coarse_torch',
    'compose_labeled_mask',
    'decompose_image_path',
    'heat_mask_path_candidates',
    'mask_path_candidates',
    'relative_data_path',
    'split_masks_from_one_mask',
    'split_masks_from_one_mask_torch',
    'split_masks_from_one_mask_with_bg',
    'train_features_path_candidates',
]

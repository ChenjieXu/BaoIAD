"""Tests for shared UniVAD asset helpers."""

import numpy as np
import torch

from baoiad.utils.univad_assets import (
    assign_fine_to_coarse_torch,
    compose_labeled_mask,
    decompose_image_path,
    heat_mask_path_candidates,
    mask_path_candidates,
    relative_data_path,
    split_masks_from_one_mask,
    split_masks_from_one_mask_with_bg,
)


def test_relative_data_path_and_decompose_image_path():
    image_path = 'data/mvtec_ad/bottle/test/broken_large/000.png'
    assert relative_data_path(image_path) == 'mvtec_ad/bottle/test/broken_large/000.png'
    assert decompose_image_path(image_path) == ('bottle', 'test', 'broken_large', '000')


def test_mask_and_heat_mask_path_candidates():
    image_path = 'data/mvtec_ad/bottle/train/good/000.png'
    mask_candidates = mask_path_candidates('/tmp/assets/masks', 'bottle', image_path)
    heat_candidates = heat_mask_path_candidates('/tmp/assets/heat_masks', 'bottle', image_path)

    assert mask_candidates[0].endswith('bottle/train/good/000.npy')
    assert any(path.endswith('mvtec_ad/bottle/train/good/000/grounding_mask.png') for path in mask_candidates)
    assert heat_candidates[0].endswith('bottle/train/good/000.png')


def test_compose_and_split_masks_roundtrip():
    masks = [
        np.array([[255, 0], [0, 0]], dtype=np.uint8),
        np.array([[0, 0], [255, 255]], dtype=np.uint8),
    ]
    labeled = compose_labeled_mask(masks)
    split_masks, split_indices = split_masks_from_one_mask(labeled)
    split_with_bg, split_with_bg_indices = split_masks_from_one_mask_with_bg(labeled)

    assert labeled.tolist() == [[1, 0], [2, 2]]
    assert split_indices == [1, 2]
    assert split_masks[0].dtype == np.uint8
    assert split_with_bg_indices == [0, 1, 2]
    assert split_with_bg[0].shape == labeled.shape


def test_assign_fine_to_coarse_torch():
    coarse = torch.tensor([
        [[1, 1], [0, 0]],
        [[0, 0], [1, 1]],
    ], dtype=torch.uint8)
    fine = torch.tensor([
        [[1, 1], [0, 0]],
        [[0, 0], [1, 1]],
    ], dtype=torch.uint8)

    assigned = assign_fine_to_coarse_torch(coarse, fine)
    assert assigned.shape == coarse.shape
    assert assigned[0, 0, 0].item() == 1
    assert assigned[1, 1, 0].item() == 2

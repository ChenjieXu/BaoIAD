"""Shared utilities for the strict GLASS alignment path.

This module ports the official GLASS helpers that are shared by the
dataset-side augmentation path and the detector-side distribution logic.
"""

from __future__ import annotations

import glob
import logging
import math
import os
import os.path as osp
import socket
import tarfile
import urllib.request
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from baoiad.utils.dtd import download_dtd as _download_dtd

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def ensure_imgaug_numpy_compat() -> None:
    """Provide the NumPy and collections shims required by imgaug on modern Python."""
    import collections
    import collections.abc

    # collections shims for Python 3.10+ (deprecated in 3.9, removed in 3.10)
    if not hasattr(collections, 'Iterable'):
        collections.Iterable = collections.abc.Iterable  # type: ignore[attr-defined]
    if not hasattr(collections, 'Mapping'):
        collections.Mapping = collections.abc.Mapping  # type: ignore[attr-defined]
    if not hasattr(collections, 'MutableMapping'):
        collections.MutableMapping = collections.abc.MutableMapping  # type: ignore[attr-defined]
    if not hasattr(collections, 'Sequence'):
        collections.Sequence = collections.abc.Sequence  # type: ignore[attr-defined]

    # NumPy shims for NumPy 2.0
    if not hasattr(np, 'sctypes'):
        np.sctypes = {  # type: ignore[attr-defined]
            'int': [np.int8, np.int16, np.int32, np.int64],
            'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
            'float': [np.float16, np.float32, np.float64],
            'complex': [np.complex64, np.complex128],
            'others': [np.bool_, np.object_, np.str_, np.bytes_],
        }


def resolve_dtd_texture_paths(dtd_path: Optional[str]) -> list[str]:
    """Resolve DTD texture image paths for GLASS-style augmentation."""
    effective_dtd_path = dtd_path
    if dtd_path == 'auto':
        effective_dtd_path = _download_dtd()

    if not effective_dtd_path:
        return []

    if osp.basename(effective_dtd_path) == 'images':
        search_root = effective_dtd_path
    elif osp.isdir(osp.join(effective_dtd_path, 'images')):
        search_root = osp.join(effective_dtd_path, 'images')
    elif osp.isdir(osp.join(effective_dtd_path, 'dtd', 'images')):
        search_root = osp.join(effective_dtd_path, 'dtd', 'images')
    else:
        search_root = effective_dtd_path

    if not osp.isdir(search_root):
        raise FileNotFoundError(f'DTD path does not exist: {search_root}')

    texture_paths = sorted(
        glob.glob(osp.join(search_root, '*', '*.jpg'))
        + glob.glob(osp.join(search_root, '*', '*.png'))
        + glob.glob(osp.join(search_root, '*.jpg'))
        + glob.glob(osp.join(search_root, '*.png'))
    )
    if not texture_paths:
        raise FileNotFoundError(f'No DTD textures found under {search_root}')
    return texture_paths


def build_rotation_augmenter():
    """Build the imgaug rotation augmenter used by the official Perlin helper."""
    ensure_imgaug_numpy_compat()
    import imgaug.augmenters as iaa

    return iaa.Sequential([iaa.Affine(rotate=(-90, 90))])


def lerp_np(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Linear interpolation helper for Perlin noise."""
    return (y - x) * w + x


def rand_perlin_2d_np(
    shape: Tuple[int, int],
    res: Tuple[int, int],
    fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3,
) -> np.ndarray:
    """Generate 2D Perlin noise using the official GLASS helper."""
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)

    def tile_grads(slice1, slice2):
        return np.repeat(
            np.repeat(gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]], d[0], axis=0),
            d[1],
            axis=1,
        )

    def dot(grad, shift):
        grid_shifted = np.stack(
            (
                grid[:shape[0], :shape[1], 0] + shift[0],
                grid[:shape[0], :shape[1], 1] + shift[1],
            ),
            axis=-1,
        )
        return (grid_shifted * grad[:shape[0], :shape[1]]).sum(axis=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * lerp_np(
        lerp_np(n00, n10, t[..., 0]),
        lerp_np(n01, n11, t[..., 0]),
        t[..., 1],
    )


def generate_thr(img_shape: Sequence[int], min_scale: int = 0, max_scale: int = 4) -> np.ndarray:
    """Generate one rotated binary Perlin mask as in the official GLASS code."""
    perlin_scalex = 2 ** np.random.randint(min_scale, max_scale)
    perlin_scaley = 2 ** np.random.randint(min_scale, max_scale)
    perlin_noise = rand_perlin_2d_np((img_shape[1], img_shape[2]), (perlin_scalex, perlin_scaley))
    rot = build_rotation_augmenter()
    perlin_noise = rot(image=perlin_noise.astype(np.float32))
    return np.where(perlin_noise > 0.5, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))


def generate_glass_perlin_masks(
    img_shape: Sequence[int],
    feat_size: int,
    min_scale: int,
    max_scale: int,
    mask_fg: torch.Tensor,
    return_large_mask: bool = True,
    max_tries: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate the official GLASS pair of small/large masks.

    Args:
        img_shape: Image tensor shape in ``(C, H, W)`` form.
        feat_size: Feature-space spatial size for ``mask_s``.
        min_scale: Minimum Perlin log-scale.
        max_scale: Maximum Perlin log-scale.
        mask_fg: Foreground mask in image space.
        return_large_mask: Whether to also return the image-space mask.
        max_tries: Safety cap to avoid infinite loops with empty foreground masks.
    """
    if mask_fg.ndim == 3:
        mask_fg = mask_fg.squeeze(0)
    mask_fg = mask_fg.float()
    if float(mask_fg.max().item()) <= 0:
        raise ValueError('Foreground mask is empty; GLASS strict augmentation cannot synthesize anomalies.')

    down_ratio_y = int(img_shape[1] / feat_size)
    down_ratio_x = int(img_shape[2] / feat_size)
    if down_ratio_y <= 0 or down_ratio_x <= 0:
        raise ValueError(
            f'Invalid GLASS downsampling ratio for img_shape={tuple(img_shape)} and feat_size={feat_size}.'
        )

    mask_s = np.zeros((feat_size, feat_size), dtype=np.float32)
    mask_l = None
    tries = 0
    while np.max(mask_s) == 0:
        tries += 1
        if tries > max_tries:
            raise RuntimeError('Failed to generate a non-empty GLASS Perlin mask after repeated attempts.')

        perlin_thr_1 = generate_thr(img_shape, min_scale=min_scale, max_scale=max_scale)
        perlin_thr_2 = generate_thr(img_shape, min_scale=min_scale, max_scale=max_scale)
        temp = float(torch.rand(1).item())
        if temp > 2 / 3:
            perlin_thr = np.where(perlin_thr_1 + perlin_thr_2 > 0, 1.0, 0.0)
        elif temp > 1 / 3:
            perlin_thr = perlin_thr_1 * perlin_thr_2
        else:
            perlin_thr = perlin_thr_1

        perlin_thr_t = torch.from_numpy(perlin_thr).float()
        perlin_thr_fg = perlin_thr_t * mask_fg
        pooled = torch.nn.functional.max_pool2d(
            perlin_thr_fg.unsqueeze(0).unsqueeze(0),
            (down_ratio_y, down_ratio_x),
        ).float()
        mask_s = pooled.cpu().numpy()[0, 0]
        mask_l = perlin_thr_fg.cpu().numpy()

    if not return_large_mask:
        return mask_s, np.zeros_like(mask_l)
    assert mask_l is not None
    return mask_s.astype(np.float32), mask_l.astype(np.float32)


def tensor_to_bgr_image(image: torch.Tensor) -> np.ndarray:
    """Convert a normalized CHW tensor back to a BGR uint8 image."""
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f'Expected a CHW RGB tensor, got shape={tuple(image.shape)}')

    mean = torch.tensor(IMAGENET_MEAN, device=image.device, dtype=image.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=image.device, dtype=image.dtype).view(3, 1, 1)
    rgb = image.detach().cpu() * std.cpu() + mean.cpu()
    rgb = rgb.clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    bgr = (rgb[:, :, ::-1] * 255.0).astype(np.uint8)
    return bgr


def distribution_judge(img_bgr: np.ndarray) -> int:
    """Judge whether the category follows the manifold or hypersphere branch."""
    resized = cv2.resize(img_bgr, (289, 289))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (39, 39))

    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1e-8)
    magnitude[magnitude > 170] = 255
    magnitude[magnitude <= 170] = 0

    height, width = magnitude.shape
    center = (height // 2, width // 2)
    y_indices, x_indices = np.where(magnitude == 255)
    if len(y_indices) == 0:
        return 0

    y_all, x_all = np.indices((2 * height, 2 * width))

    l1_dist_x = np.abs(x_indices - center[1])
    l1_dist_y = np.abs(y_indices - center[0])
    dist = np.sqrt((x_indices - center[1]) ** 2 + (y_indices - center[0]) ** 2)
    l2_dist_all = np.sqrt((x_all - center[1]) ** 2 + (y_all - center[0]) ** 2)

    side_x = np.max(l1_dist_x)
    side_y = np.max(l1_dist_y)
    radius = np.max(dist)
    points_num = len(dist)

    l1_density = points_num / (4 * max(side_x, 1) * max(side_y, 1))
    l2_density = points_num / (np.sum(l2_dist_all <= radius) + 1e-10)
    return int((l1_density > 0.21 or l2_density > 0.21) and radius > 12 and points_num > 60)

"""DeSTSeg-specific data synthesis and packing transforms."""

from __future__ import annotations

import glob
import math
import os
from typing import Dict, Iterable, Sequence

import cv2
import numpy as np
import torch
from mmcv.transforms import BaseTransform
from PIL import Image

from baoiad.registry import TRANSFORMS
from baoiad.structures import ADDataSample


_IMAGENET_MEAN = np.array((123.675, 116.28, 103.53), dtype=np.float32)
_IMAGENET_STD = np.array((58.395, 57.12, 57.375), dtype=np.float32)
_NO_ROTATION_CATEGORIES = {'capsule', 'metal_nut', 'pill', 'toothbrush', 'transistor'}
_SLIGHT_ROTATION_CATEGORIES = {'wood', 'zipper', 'cable'}
_ROTATION_CATEGORIES = {'bottle', 'grid', 'hazelnut', 'leather', 'tile', 'carpet', 'screw'}


def _lerp_np(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    return (y - x) * w + x


def _rand_perlin_2d_np(
    shape: Sequence[int],
    res: Sequence[int],
    fade=lambda t: 6 * t**5 - 15 * t**4 + 10 * t**3,
) -> np.ndarray:
    """Generate 2D Perlin noise matching the official implementation."""
    h, w = shape
    delta = (res[0] / h, res[1] / w)
    d = (max(1, h // res[0]), max(1, w // res[1]))
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    tile_grads = lambda s1, s2: cv2.resize(  # noqa: E731
        np.repeat(
            np.repeat(gradients[s1[0]:s1[1], s2[0]:s2[1]], d[0], axis=0),
            d[1],
            axis=1,
        ),
        dsize=(w, h),
        interpolation=cv2.INTER_LINEAR,
    )
    dot = lambda grad, shift: (  # noqa: E731
        np.stack((grid[:h, :w, 0] + shift[0], grid[:h, :w, 1] + shift[1]), axis=-1)
        * grad[:h, :w]
    ).sum(axis=-1)
    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:h, :w])
    return math.sqrt(2) * _lerp_np(
        _lerp_np(n00, n10, t[..., 0]),
        _lerp_np(n01, n11, t[..., 0]),
        t[..., 1],
    )


def _apply_category_rotation(image: Image.Image, cls_name: str | None) -> Image.Image:
    fill_color = (114, 114, 114)
    if cls_name in _ROTATION_CATEGORIES:
        degree = int(np.random.choice(np.array([0, 90, 180, 270])))
        image = image.rotate(degree, fillcolor=fill_color, resample=Image.BILINEAR)
    if cls_name in _SLIGHT_ROTATION_CATEGORIES or cls_name in _ROTATION_CATEGORIES:
        degree = float(np.random.uniform(-5.0, 5.0))
        image = image.rotate(degree, fillcolor=fill_color, resample=Image.BILINEAR)
    return image


def _rotate_array(image: np.ndarray, angle: float, fill_value=0):
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill_value,
    )


def _normalize_imagenet(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    if image.max() <= 1.0:
        image = image * 255.0
    return (image - _IMAGENET_MEAN) / _IMAGENET_STD


def resolve_destseg_dtd_images_dir(dtd_path: str) -> str:
    """Resolve the actual DTD image directory from a strict config path."""
    candidates: list[str] = []
    if dtd_path == 'auto':
        candidates.extend([
            os.path.join('data', 'dtd', 'images'),
            os.path.join('data', 'dtd', 'dtd', 'images'),
        ])
    elif dtd_path:
        candidates.extend([
            dtd_path,
            os.path.join(dtd_path, 'images'),
            os.path.join(dtd_path, 'dtd', 'images'),
        ])

    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(
        f'Unable to locate DTD images directory for dtd_path={dtd_path!r}. '
        'Expected an images directory such as data/dtd/dtd/images.'
    )


def _collect_texture_paths(images_dir: str) -> list[str]:
    patterns = ('*.jpg', '*.png', '*.jpeg', '*.bmp')
    paths: list[str] = []
    for pattern in patterns:
        paths.extend(glob.glob(os.path.join(images_dir, '*', pattern)))
        paths.extend(glob.glob(os.path.join(images_dir, pattern)))
    return sorted(set(paths))


@TRANSFORMS.register_module()
class DeSTSegAugment(BaseTransform):
    """Synthesize official-style DeSTSeg training pairs in image space."""

    def __init__(self, dtd_path: str = 'auto', perlin_scale: int = 6, beta_max: float = 0.8) -> None:
        self.perlin_scale = perlin_scale
        self.beta_max = beta_max
        images_dir = resolve_destseg_dtd_images_dir(dtd_path)
        self.texture_paths = _collect_texture_paths(images_dir)
        if not self.texture_paths:
            raise FileNotFoundError(f'No DTD texture images found under {images_dir}')

    def _sample_texture(self, size: tuple[int, int]) -> np.ndarray:
        index = torch.randint(0, len(self.texture_paths), (1,)).item()
        path = self.texture_paths[index]
        texture = Image.open(path).convert('RGB')
        texture = texture.resize(size, Image.BILINEAR)
        return np.array(texture, dtype=np.float32)

    def transform(self, results: Dict) -> Dict:
        img = results['img']
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        cls_name = results.get('cls_name', '')
        image = Image.fromarray(img)
        image = _apply_category_rotation(image, cls_name)
        clean = np.array(image, dtype=np.float32)
        h, w = clean.shape[:2]

        scalex = 2 ** torch.randint(0, self.perlin_scale, (1,)).item()
        scaley = 2 ** torch.randint(0, self.perlin_scale, (1,)).item()
        noise = _rand_perlin_2d_np((h, w), (scalex, scaley)).astype(np.float32)
        angle = float(torch.empty(1).uniform_(-90.0, 90.0).item())
        noise = _rotate_array(noise, angle, fill_value=0)
        noise = np.expand_dims(noise, axis=2)
        mask = np.where(noise > 0.5, np.ones_like(noise), np.zeros_like(noise)).astype(np.float32)

        texture = self._sample_texture((w, h))
        clean_f = clean.astype(np.float32) / 255.0
        texture_f = texture.astype(np.float32) / 255.0
        beta = float(torch.rand(1).item()) * self.beta_max
        augmented = clean_f * (1.0 - mask) + (1.0 - beta) * texture_f * mask + beta * clean_f * mask

        results['img'] = _normalize_imagenet(augmented)
        results['img_origin'] = _normalize_imagenet(clean_f)
        results['img_aug'] = results['img']
        results['gt_mask'] = mask[..., 0].astype(np.float32)
        results['gt_label'] = int(mask.max() > 0)
        return results


@TRANSFORMS.register_module()
class PackDeSTSegInputs(BaseTransform):
    """Pack DeSTSeg training pairs with explicit clean/augmented tensors."""

    def transform(self, results: Dict) -> Dict:
        def _to_chw_tensor(value: np.ndarray | torch.Tensor) -> torch.Tensor:
            if isinstance(value, np.ndarray):
                if value.ndim == 2:
                    value = value[..., np.newaxis]
                if value.ndim == 3 and value.shape[-1] <= 4:
                    value = value.transpose(2, 0, 1)
                value = torch.from_numpy(value).contiguous().float()
            return value

        img_aug = _to_chw_tensor(results['img'])
        img_origin = _to_chw_tensor(results['img_origin'])
        gt_mask = _to_chw_tensor(results['gt_mask']).squeeze(0)

        data_sample = ADDataSample()
        data_sample.set_metainfo({
            'cls_name': results.get('cls_name', ''),
            'img_path': results.get('img_path', ''),
            'defect_type': results.get('defect_type', ''),
            'img_origin': img_origin,
            'img_aug': img_aug,
        })
        data_sample.gt_label = results.get('gt_label', 0)
        data_sample.gt_mask = gt_mask

        return dict(inputs=img_aug, data_samples=data_sample)

"""Strict official preprocessing helpers for CFlow."""

from __future__ import annotations

from typing import Dict

import numpy as np
from mmcv.transforms import BaseTransform
from PIL import Image
from torchvision.transforms import CenterCrop, InterpolationMode, RandomRotation, Resize

from baoiad.registry import TRANSFORMS


@TRANSFORMS.register_module()
class CFlowOfficialTransform(BaseTransform):
    """Apply the official CFlow resize / rotate / crop policy."""

    def __init__(
        self,
        size_map: Dict[str, int],
        default_size: int = 256,
        train: bool = False,
    ) -> None:
        self.size_map = {str(key): int(value) for key, value in size_map.items()}
        self.default_size = int(default_size)
        self.train = bool(train)
        self._image_ops: dict[int, list] = {}
        self._mask_ops: dict[int, list] = {}

    def _resolve_size(self, results: Dict) -> int:
        cls_name = str(results.get('cls_name', ''))
        return int(self.size_map.get(cls_name, self.default_size))

    def _build_image_ops(self, size: int) -> list:
        ops = [
            Resize(size, interpolation=InterpolationMode.BILINEAR, antialias=True),
        ]
        if self.train:
            ops.append(RandomRotation(5))
        ops.append(CenterCrop(size))
        return ops

    def _build_mask_ops(self, size: int) -> list:
        ops = [
            Resize(size, interpolation=InterpolationMode.NEAREST, antialias=False),
        ]
        if self.train:
            ops.append(
                RandomRotation(
                    5,
                    interpolation=InterpolationMode.NEAREST,
                    fill=0,
                )
            )
        ops.append(CenterCrop(size))
        return ops

    def _apply_ops(self, image: Image.Image, ops: list) -> Image.Image:
        output = image
        for op in ops:
            output = op(output)
        return output

    def _get_image_ops(self, size: int) -> list:
        if size not in self._image_ops:
            self._image_ops[size] = self._build_image_ops(size)
        return self._image_ops[size]

    def _get_mask_ops(self, size: int) -> list:
        if size not in self._mask_ops:
            self._mask_ops[size] = self._build_mask_ops(size)
        return self._mask_ops[size]

    def transform(self, results: Dict) -> Dict:
        size = self._resolve_size(results)

        img = results['img']
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img)
        pil_img = self._apply_ops(pil_img, self._get_image_ops(size))
        results['img'] = np.asarray(pil_img, dtype=np.float32)
        results['img_shape'] = results['img'].shape[:2]

        if 'gt_mask' in results:
            mask = results['gt_mask']
            mask = (mask > 0).astype(np.uint8) * 255
            pil_mask = Image.fromarray(mask)
            pil_mask = self._apply_ops(pil_mask, self._get_mask_ops(size))
            results['gt_mask'] = (np.asarray(pil_mask) > 0).astype(np.float32)

        return results

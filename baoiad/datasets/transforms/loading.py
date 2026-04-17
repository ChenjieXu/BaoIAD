"""Image and mask loading transforms."""

from typing import Dict

import cv2
import numpy as np
from mmcv.transforms import BaseTransform
from PIL import Image

from baoiad.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadImage(BaseTransform):
    """Load image from file path.

    Required keys: img_path
    Added keys: img, img_shape, ori_shape
    """

    def __init__(
        self,
        color_type: str = 'color',
        to_float32: bool = False,
        to_rgb: bool = True,
        keep_bgr_copy: bool = False,
        backend: str = 'cv2',
    ) -> None:
        self.color_type = color_type
        self.to_float32 = to_float32
        self.to_rgb = to_rgb
        self.keep_bgr_copy = keep_bgr_copy
        self.backend = backend

    def transform(self, results: Dict) -> Dict:
        img_path = results['img_path']
        if self.backend == 'pil':
            if self.color_type == 'color':
                pil_img = Image.open(img_path).convert('RGB')
                img = np.asarray(pil_img)
                if not self.to_rgb:
                    img = img[..., ::-1]
                if self.keep_bgr_copy:
                    results['ori_img_bgr'] = img[..., ::-1].copy() if self.to_rgb else img.copy()
            else:
                pil_img = Image.open(img_path).convert('L')
                img = np.asarray(pil_img)
        else:
            flag = cv2.IMREAD_COLOR if self.color_type == 'color' else cv2.IMREAD_GRAYSCALE
            img = cv2.imread(img_path, flag)
            if img is None:
                raise FileNotFoundError(f'Failed to load image: {img_path}')
            if self.color_type == 'color':
                if self.keep_bgr_copy:
                    results['ori_img_bgr'] = img.copy()
                if self.to_rgb:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img is None:
            raise FileNotFoundError(f'Failed to load image: {img_path}')
        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results


@TRANSFORMS.register_module()
class LoadMask(BaseTransform):
    """Load ground truth mask for anomaly segmentation.

    Required keys: gt_mask_path
    Added keys: gt_mask
    """

    def __init__(self, backend: str = 'cv2', to_binary: bool = True) -> None:
        self.backend = backend
        self.to_binary = to_binary

    def transform(self, results: Dict) -> Dict:
        mask_path = results.get('gt_mask_path', '')
        if mask_path:
            if self.backend == 'pil':
                mask = np.asarray(Image.open(mask_path).convert('L'))
            else:
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                if self.to_binary:
                    mask = (mask > 0).astype(np.float32)
                else:
                    mask = mask.astype(np.float32) / 255.0
            else:
                mask = np.zeros(results['img_shape'][:2], dtype=np.float32)
        else:
            # Normal image: all-zero mask
            shape = results['img_shape'] if 'img_shape' in results else results['img'].shape[:2]
            mask = np.zeros(shape, dtype=np.float32)

        results['gt_mask'] = mask
        return results

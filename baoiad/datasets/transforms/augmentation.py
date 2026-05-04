"""Resize and normalize transforms for anomaly detection."""

from typing import Dict, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from mmcv.transforms import BaseTransform
from PIL import Image

from baoiad.registry import TRANSFORMS
from baoiad.utils.rdpp_noise import Simplex_CLASS


@TRANSFORMS.register_module(force=True)
class ResizeAD(BaseTransform):
    """Resize image and mask to a fixed size.

    Args:
        size: Target (height, width).
        interpolation: OpenCV interpolation flag (ignored when backend='pillow').
        backend: 'cv2' or 'pillow'. Use 'pillow' to match anomalib/torchvision
            preprocessing (PIL load + torchvision Resize with antialias).
    """

    def __init__(
        self,
        size: Union[int, Tuple[int, int]] = 256,
        interpolation: int = cv2.INTER_LINEAR,
        backend: str = 'pillow',
        mask_interpolation: str = 'nearest',
        keep_ratio: bool = False,
        official_pil: bool = False,
    ) -> None:
        self.size = size
        self.interpolation = interpolation
        self.backend = backend
        self.mask_interpolation = mask_interpolation
        self.keep_ratio = keep_ratio
        self.official_pil = bool(official_pil)
        if self.mask_interpolation not in ('nearest', 'bilinear'):
            raise ValueError("mask_interpolation must be 'nearest' or 'bilinear'.")
        if backend == 'pillow':
            from torchvision.transforms import Resize as TVResize
            from torchvision.transforms import InterpolationMode

            # When keep_ratio=False, pass (H, W) tuple so torchvision
            # resizes to a fixed square instead of keeping aspect ratio.
            _tv_size = size if (isinstance(size, tuple) or keep_ratio) else (size, size)

            self._tv_resize = TVResize(
                _tv_size,
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            self._tv_mask_resize = TVResize(
                _tv_size,
                interpolation=(
                    InterpolationMode.NEAREST
                    if self.mask_interpolation == 'nearest'
                    else InterpolationMode.BILINEAR
                ),
                antialias=False if self.mask_interpolation == 'nearest' else True,
            )

    def _target_hw(self, img: np.ndarray) -> Tuple[int, int]:
        if isinstance(self.size, tuple):
            return self.size

        if not self.keep_ratio:
            return (self.size, self.size)

        h, w = img.shape[:2]
        if h <= w:
            scale = self.size / float(h)
        else:
            scale = self.size / float(w)
        return (int(round(h * scale)), int(round(w * scale)))

    def transform(self, results: Dict) -> Dict:
        out_h, out_w = self._target_hw(results['img'])

        if self.backend == 'pillow':
            img = results['img']
            if self.official_pil:
                if img.dtype != np.uint8:
                    img = img.astype(np.float32)
                    if img.max() <= 1.0:
                        img = img * 255.0
                    img = np.clip(img, 0.0, 255.0).astype(np.uint8)
                pil_img = Image.fromarray(img)
                resized = self._tv_resize(pil_img)
                results['img'] = np.asarray(resized)
            else:
                import torch

                pil_img = Image.fromarray(img if img.dtype == np.uint8 else img.astype(np.uint8))
                tensor = torch.from_numpy(np.array(pil_img))
                is_grayscale = tensor.ndim == 2
                if is_grayscale:
                    tensor = tensor.unsqueeze(0)
                else:
                    tensor = tensor.permute(2, 0, 1)
                tensor = tensor.float() / 255.0
                tensor = self._tv_resize(tensor)
                if is_grayscale:
                    results['img'] = tensor.squeeze(0).numpy() * 255.0
                else:
                    results['img'] = tensor.permute(1, 2, 0).numpy() * 255.0
        else:
            results['img'] = cv2.resize(
                results['img'],
                (out_w, out_h),
                interpolation=self.interpolation,
            )

        results['img_shape'] = results['img'].shape[:2]

        if 'gt_mask' in results:
            if self.backend == 'pillow':
                mask = results['gt_mask'].astype(np.float32)
                if self.official_pil:
                    mask = np.clip(mask, 0.0, 1.0)
                    mask_img = Image.fromarray((mask * 255.0).astype(np.uint8), mode='L')
                    resized = self._tv_mask_resize(mask_img)
                    results['gt_mask'] = np.asarray(resized).astype(np.float32) / 255.0
                else:
                    import torch

                    mask_tensor = torch.from_numpy(mask).unsqueeze(0)
                    mask_tensor = self._tv_mask_resize(mask_tensor)
                    results['gt_mask'] = mask_tensor.squeeze(0).numpy()
            else:
                results['gt_mask'] = cv2.resize(
                    results['gt_mask'],
                    (out_w, out_h),
                    interpolation=(
                        cv2.INTER_NEAREST
                        if self.mask_interpolation == 'nearest'
                        else cv2.INTER_LINEAR
                    ),
                )
        return results


@TRANSFORMS.register_module(force=True)
class ThresholdMask(BaseTransform):
    """Threshold a float mask into a binary mask."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = float(threshold)

    def transform(self, results: Dict) -> Dict:
        if 'gt_mask' not in results:
            return results
        mask = results['gt_mask'].astype(np.float32)
        results['gt_mask'] = np.where(mask < self.threshold, 0.0, 1.0).astype(np.float32)
        return results


@TRANSFORMS.register_module(force=True)
class RandomRotation(BaseTransform):
    """Random rotation augmentation.

    Args:
        degrees: Range of degrees to rotate. If float, range is (-degrees, +degrees).
    """

    def __init__(self, degrees: float = 180.0) -> None:
        self.degrees = degrees

    def transform(self, results: Dict) -> Dict:
        angle = np.random.uniform(-self.degrees, self.degrees)
        h, w = results['img'].shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        results['img'] = cv2.warpAffine(
            results['img'], M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        if 'gt_mask' in results:
            results['gt_mask'] = cv2.warpAffine(
                results['gt_mask'], M, (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        return results


@TRANSFORMS.register_module(force=True)
class NormalizeAD(BaseTransform):
    """Normalize image with ImageNet statistics.

    Args:
        mean: Per-channel mean.
        std: Per-channel std.
    """

    IMAGENET_MEAN = (123.675, 116.28, 103.53)
    IMAGENET_STD = (58.395, 57.12, 57.375)

    def __init__(
        self,
        mean: Tuple[float, ...] = IMAGENET_MEAN,
        std: Tuple[float, ...] = IMAGENET_STD,
        keys: Sequence[str] = ('img',),
    ) -> None:
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.keys = tuple(keys)

    def transform(self, results: Dict) -> Dict:
        for key in self.keys:
            if key not in results:
                continue
            img = results[key].astype(np.float32)
            img = (img - self.mean) / self.std
            results[key] = img
        return results


@TRANSFORMS.register_module(force=True)
class OpenCLIPPreprocessAD(BaseTransform):
    """Apply the official OpenCLIP-style val preprocessing.

    This matches the MuSc reference CLIP preprocessing path:
    `Resize((size, size), bicubic) -> CenterCrop(size) -> ToTensor() ->
    Normalize(OPENAI_MEAN, OPENAI_STD)`.

    Masks are resized and cropped with torchvision as well so MuSc's
    prediction maps and evaluation masks share the same spatial path.
    """

    OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
    OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        crop_size: Optional[Union[int, Tuple[int, int]]] = None,
        mean: Tuple[float, ...] = OPENAI_DATASET_MEAN,
        std: Tuple[float, ...] = OPENAI_DATASET_STD,
    ) -> None:
        from torchvision.transforms import CenterCrop as TVCenterCrop
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import Normalize, Resize as TVResize, ToTensor

        self.size = size if isinstance(size, tuple) else (size, size)
        self.crop_size = crop_size if crop_size is not None else self.size
        if isinstance(self.crop_size, int):
            self.crop_size = (self.crop_size, self.crop_size)
        self.mean = tuple(mean)
        self.std = tuple(std)

        self._img_resize = TVResize(
            self.size,
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        self._img_crop = TVCenterCrop(self.crop_size)
        self._mask_resize = TVResize(
            self.size,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        self._mask_crop = TVCenterCrop(self.crop_size)
        self._to_tensor = ToTensor()
        self._normalize = Normalize(mean=self.mean, std=self.std)

    @staticmethod
    def _to_uint8_image(img: np.ndarray) -> np.ndarray:
        if img.dtype == np.uint8:
            return img
        img = img.astype(np.float32)
        if img.max() <= 1.0:
            img = img * 255.0
        return np.clip(img, 0.0, 255.0).astype(np.uint8)

    @staticmethod
    def _to_uint8_mask(mask: np.ndarray) -> np.ndarray:
        if mask.dtype == np.uint8:
            return mask
        mask = mask.astype(np.float32)
        if mask.max() <= 1.0:
            mask = mask * 255.0
        return np.clip(mask, 0.0, 255.0).astype(np.uint8)

    def transform(self, results: Dict) -> Dict:
        from PIL import Image

        img = self._to_uint8_image(results['img'])
        img_tensor = self._to_tensor(self._img_crop(self._img_resize(Image.fromarray(img).convert('RGB'))))
        img_tensor = self._normalize(img_tensor)
        results['img'] = img_tensor.permute(1, 2, 0).numpy().astype(np.float32)
        results['img_shape'] = tuple(self.crop_size)

        if 'gt_mask' in results:
            mask = self._to_uint8_mask(results['gt_mask'])
            mask_tensor = self._to_tensor(self._mask_crop(self._mask_resize(Image.fromarray(mask))))
            results['gt_mask'] = mask_tensor.squeeze(0).numpy().astype(np.float32)

        return results

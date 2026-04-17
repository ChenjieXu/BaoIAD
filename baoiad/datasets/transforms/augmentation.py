"""Resize and normalize transforms for anomaly detection."""

from typing import Dict, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from mmcv.transforms import BaseTransform
from PIL import Image

from baoiad.registry import TRANSFORMS
from baoiad.utils.rdpp_noise import Simplex_CLASS


@TRANSFORMS.register_module()
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


@TRANSFORMS.register_module()
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


@TRANSFORMS.register_module()
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


@TRANSFORMS.register_module()
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


@TRANSFORMS.register_module()
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


@TRANSFORMS.register_module()
class GraphCorePreprocessAD(BaseTransform):
    """Exact GraphCore preprocessing using PIL + torchvision transforms.

    Matches open-iad's ``aug_type('normal')`` image path:
    ``Resize((224, 224)) -> CenterCrop(224) -> ToTensor() -> Normalize(ImageNet)``.
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        size: int | tuple[int, int] = 224,
        crop_size: int | tuple[int, int] | None = None,
        mask_size: int | tuple[int, int] | None = None,
        mask_crop_size: int | tuple[int, int] | None = None,
        mean: tuple[float, ...] = IMAGENET_MEAN,
        std: tuple[float, ...] = IMAGENET_STD,
    ) -> None:
        from torchvision.transforms import CenterCrop as TVCenterCrop
        from torchvision.transforms import Normalize, Resize as TVResize, ToTensor

        self.size = size if isinstance(size, tuple) else (size, size)
        self.crop_size = crop_size if crop_size is not None else self.size
        if isinstance(self.crop_size, int):
            self.crop_size = (self.crop_size, self.crop_size)
        self.mask_size = mask_size if mask_size is not None else (
            size[0] if isinstance(size, tuple) else size
        )
        self.mask_crop_size = mask_crop_size if mask_crop_size is not None else (
            crop_size if crop_size is not None else (size[0] if isinstance(size, tuple) else size)
        )
        if isinstance(self.mask_crop_size, tuple) and self.mask_crop_size[0] == self.mask_crop_size[1]:
            self.mask_crop_size = self.mask_crop_size[0]
        self.mean = tuple(mean)
        self.std = tuple(std)
        self._img_resize = TVResize(self.size)
        self._img_crop = TVCenterCrop(self.crop_size)
        self._mask_resize = TVResize(self.mask_size)
        self._mask_crop = TVCenterCrop(self.mask_crop_size)
        self._to_tensor = ToTensor()
        self._normalize = Normalize(mean=self.mean, std=self.std)

    @staticmethod
    def _mask_to_uint8(mask: np.ndarray) -> np.ndarray:
        if mask.dtype == np.uint8:
            return mask
        mask = mask.astype(np.float32)
        if mask.max() <= 1.0:
            mask = mask * 255.0
        return np.clip(mask, 0.0, 255.0).astype(np.uint8)

    def transform(self, results: Dict) -> Dict:
        from PIL import Image

        img = Image.open(results['img_path']).convert('RGB')
        img_tensor = self._to_tensor(self._img_crop(self._img_resize(img)))
        img_tensor = self._normalize(img_tensor)
        results['img'] = img_tensor.permute(1, 2, 0).numpy().astype(np.float32)
        results['img_shape'] = tuple(self.crop_size)
        results['ori_shape'] = tuple(self.crop_size)

        if 'gt_mask' in results:
            mask_img = Image.fromarray(self._mask_to_uint8(results['gt_mask']))
            mask_tensor = self._to_tensor(self._mask_crop(self._mask_resize(mask_img)))
            results['gt_mask'] = mask_tensor.squeeze(0).numpy().astype(np.float32)
        return results


@TRANSFORMS.register_module()
class ScaleNormalizeAD(BaseTransform):
    """Scale image to [0, 1] range without ImageNet normalization.

    Used by methods that require raw [0,1] images (DRAEM, DSR, etc.).
    """

    def transform(self, results: Dict) -> Dict:
        img = results['img'].astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        results['img'] = img
        return results


@TRANSFORMS.register_module()
class GenerateRDPPNoise(BaseTransform):
    """Generate the official RD++ simplex-noise image branch."""

    def __init__(
        self,
        octaves: int = 6,
        persistence: float = 0.6,
        amplitude: float = 0.2,
        min_patch_size: int = 10,
    ) -> None:
        self.octaves = int(octaves)
        self.persistence = float(persistence)
        self.amplitude = float(amplitude)
        self.min_patch_size = int(min_patch_size)
        self.simplex_noise = Simplex_CLASS()

    def transform(self, results: Dict) -> Dict:
        img = results['img'].astype(np.float32)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f'GenerateRDPPNoise expects HWC RGB image, got shape {img.shape!r}.')

        h, w, _ = img.shape
        patch_h_high = max(self.min_patch_size + 1, int(h // 8))
        patch_w_high = max(self.min_patch_size + 1, int(w // 8))
        h_noise = np.random.randint(self.min_patch_size, patch_h_high)
        w_noise = np.random.randint(self.min_patch_size, patch_w_high)

        start_h_high = max(2, h - h_noise)
        start_w_high = max(2, w - w_noise)
        start_h = np.random.randint(1, start_h_high)
        start_w = np.random.randint(1, start_w_high)

        simplex_noise = self.simplex_noise.rand_3d_octaves(
            (3, h_noise, w_noise),
            self.octaves,
            self.persistence,
        ).transpose(1, 2, 0).astype(np.float32)

        img_noise = img.copy()
        img_noise[
            start_h:start_h + h_noise,
            start_w:start_w + w_noise,
            :,
        ] += self.amplitude * simplex_noise

        results['img'] = img
        results['img_noise'] = img_noise
        return results


@TRANSFORMS.register_module()
class CenterCrop(BaseTransform):
    """Center crop image and mask to a fixed size.

    Args:
        size: Target (height, width) or single int for square crop.
    """

    def __init__(self, size: Union[int, Tuple[int, int]]) -> None:
        if isinstance(size, int):
            size = (size, size)
        self.size = size

    def transform(self, results: Dict) -> Dict:
        h, w = results['img'].shape[:2]
        th, tw = self.size

        # Calculate crop coordinates
        i = (h - th) // 2
        j = (w - tw) // 2

        results['img'] = results['img'][i:i + th, j:j + tw]
        if 'gt_mask' in results:
            results['gt_mask'] = results['gt_mask'][i:i + th, j:j + tw]

        results['img_shape'] = self.size
        return results


@TRANSFORMS.register_module()
class RandomCrop(BaseTransform):
    """Random crop image and mask to a fixed size.

    Args:
        size: Target (height, width) or single int for square crop.
    """

    def __init__(self, size: Union[int, Tuple[int, int]]) -> None:
        if isinstance(size, int):
            size = (size, size)
        self.size = size

    def transform(self, results: Dict) -> Dict:
        h, w = results['img'].shape[:2]
        th, tw = self.size

        # Handle case where target size is larger than image
        if th >= h or tw >= w:
            # Just return as-is (will be handled by Resize if needed)
            results['img_shape'] = (h, w)
            return results

        # Random crop coordinates
        i = np.random.randint(0, h - th + 1)
        j = np.random.randint(0, w - tw + 1)

        results['img'] = results['img'][i:i + th, j:j + tw]
        if 'gt_mask' in results:
            results['gt_mask'] = results['gt_mask'][i:i + th, j:j + tw]

        results['img_shape'] = self.size
        return results


@TRANSFORMS.register_module()
class RandomVerticalFlip(BaseTransform):
    """Random vertical flip with probability p.

    Args:
        p: Probability of flip.
    """

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def transform(self, results: Dict) -> Dict:
        if np.random.random() < self.p:
            results['img'] = np.flipud(results['img']).copy()
            if 'gt_mask' in results:
                results['gt_mask'] = np.flipud(results['gt_mask']).copy()
        return results


@TRANSFORMS.register_module()
class RandomHorizontalFlip(BaseTransform):
    """Random horizontal flip with probability p."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def transform(self, results: Dict) -> Dict:
        if np.random.random() < self.p:
            results['img'] = np.fliplr(results['img']).copy()
            if 'gt_mask' in results:
                results['gt_mask'] = np.fliplr(results['gt_mask']).copy()
        return results


@TRANSFORMS.register_module()
class PyramidFlowStrictTrainTransform(BaseTransform):
    """Paper-aligned texture augmentation for PyramidFlow strict training.

    The paper states that textural classes use flips and rotations with
    probability 0.5, while object classes do not use augmentation.
    """

    TEXTURES = {'carpet', 'grid', 'leather', 'tile', 'wood'}

    def __init__(
        self,
        flip_p: float = 0.5,
        rotation_p: float = 0.5,
        rotation_degrees: float = 180.0,
    ) -> None:
        self.flip_p = flip_p
        self.rotation_p = rotation_p
        self.random_horizontal_flip = RandomHorizontalFlip(p=1.0)
        self.random_rotation = RandomRotation(degrees=rotation_degrees)

    def transform(self, results: Dict) -> Dict:
        cls_name = results.get('cls_name', None)
        if cls_name not in self.TEXTURES:
            return results

        if np.random.random() < self.flip_p:
            results = self.random_horizontal_flip.transform(results)
        if np.random.random() < self.rotation_p:
            results = self.random_rotation.transform(results)
        return results


@TRANSFORMS.register_module()
class NSATransform(BaseTransform):
    """Category-aware transform for NSA training.

    Applies different transforms based on category type:
    - Unaligned objects (bottle, hazelnut, metal_nut, screw):
      RandomRotation(5) + CenterCrop(230) + RandomCrop(224)
    - Aligned objects (cable, capsule, pill, transistor, toothbrush, zipper):
      CenterCrop(230) + RandomCrop(224)
    - Textures (carpet, grid, leather, tile, wood):
      Resize(264) + RandomVerticalFlip + RandomCrop(256)

    This matches the reference implementation from train_mvtec.py.
    Note: Original NSA uses res=264 for textures, then RandomCrop(256).
    """

    UNALIGNED_OBJECTS = {'bottle', 'hazelnut', 'metal_nut', 'screw'}
    ALIGNED_OBJECTS = {'cable', 'capsule', 'pill', 'transistor', 'toothbrush', 'zipper'}
    TEXTURES = {'carpet', 'grid', 'leather', 'tile', 'wood'}

    def __init__(self) -> None:
        self.random_rotation = RandomRotation(degrees=5)
        self.center_crop_230 = CenterCrop(size=230)
        self.random_crop_224 = RandomCrop(size=224)
        self.random_crop_256 = RandomCrop(size=256)
        self.random_vertical_flip = RandomVerticalFlip(p=0.5)
        # Textures need resize to 264 before RandomCrop(256) (original uses res=264)
        self.resize_264 = ResizeAD(size=264)

    def transform(self, results: Dict) -> Dict:
        cls_name = results.get('cls_name', None)

        if cls_name in self.UNALIGNED_OBJECTS:
            # Unaligned objects: RandomRotation + CenterCrop + RandomCrop(224)
            results = self.random_rotation.transform(results)
            results = self.center_crop_230.transform(results)
            results = self.random_crop_224.transform(results)
        elif cls_name in self.ALIGNED_OBJECTS:
            # Aligned objects: CenterCrop + RandomCrop(224)
            results = self.center_crop_230.transform(results)
            results = self.random_crop_224.transform(results)
        elif cls_name in self.TEXTURES:
            # Textures: Resize(264) + RandomVerticalFlip + RandomCrop(256)
            # Original NSA uses res=264 for textures, then RandomCrop(256)
            results = self.resize_264.transform(results)
            results = self.random_vertical_flip.transform(results)
            results = self.random_crop_256.transform(results)
        else:
            # Default: use 224x224 pipeline
            results = self.center_crop_230.transform(results)
            results = self.random_crop_224.transform(results)

        return results


@TRANSFORMS.register_module()
class NSATestTransform(BaseTransform):
    """Category-aware test transform for NSA.

    - Objects: Resize to 224x224
    - Textures: Resize to 256x256
    """

    TEXTURES = {'carpet', 'grid', 'leather', 'tile', 'wood'}

    def transform(self, results: Dict) -> Dict:
        cls_name = results.get('cls_name', None)
        size = 256 if cls_name in self.TEXTURES else 224

        h, w = size, size
        results['img'] = cv2.resize(results['img'], (w, h), interpolation=cv2.INTER_LINEAR)
        results['img_shape'] = (h, w)

        if 'gt_mask' in results:
            results['gt_mask'] = cv2.resize(
                results['gt_mask'], (w, h), interpolation=cv2.INTER_NEAREST,
            )
        return results

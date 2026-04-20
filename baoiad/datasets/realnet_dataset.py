"""RealNet-specific training dataset.

This dataset follows the official RealNet MVTec training path:
- load normal training images
- sample anomaly type from configured probabilities
- synthesize SDAS/DTD anomalies with Perlin masks
- normalize both anomaly image and clean image with ImageNet statistics
"""

from __future__ import annotations

import glob
import logging
import math
import os
import tarfile
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from mmengine.dataset import BaseDataset
from PIL import Image, ImageEnhance, ImageOps
from skimage import morphology

from baoiad.registry import DATASETS
from baoiad.utils.compat import ensure_legacy_imgaug_compat
from baoiad.utils.dtd import download_dtd as _download_dtd

logger = logging.getLogger(__name__)

def _import_imgaug():
    ensure_legacy_imgaug_compat()
    import imgaug.augmenters as iaa

    return iaa


def _get_imgaug_augmenters():
    """Return the official RealNet texture augmenters.

    The reference implementation uses a 10-op imgaug pool. Our runtime ships
    imgaug 0.3.0, which lacks several of those operators, so we provide
    lightweight local implementations that preserve the official operator set
    and random-choice semantics.
    """

    def _as_uint8(image: np.ndarray) -> np.ndarray:
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    class _CallableAugmenter:
        def __init__(self, fn):
            self.fn = fn

        def __call__(self, image=None, **kwargs):
            if image is None:
                image = kwargs['image']
            return self.fn(np.asarray(image))

    def _gamma_contrast(image: np.ndarray) -> np.ndarray:
        image = _as_uint8(image)
        gamma = np.random.uniform(0.5, 2.0, size=3).astype(np.float32)
        normalized = np.clip(image.astype(np.float32) / 255.0, 0.0, 1.0)
        adjusted = np.power(normalized, gamma.reshape(1, 1, 3))
        return np.clip(adjusted * 255.0, 0.0, 255.0).astype(np.uint8)

    def _multiply_add_brightness(image: np.ndarray) -> np.ndarray:
        image = _as_uint8(image).astype(np.float32)
        mul = float(np.random.uniform(0.8, 1.2))
        add = float(np.random.uniform(-30.0, 30.0))
        return np.clip(image * mul + add, 0.0, 255.0).astype(np.uint8)

    def _enhance_sharpness(image: np.ndarray) -> np.ndarray:
        pil_img = Image.fromarray(_as_uint8(image))
        factor = float(np.random.uniform(0.0, 2.0))
        return np.asarray(ImageEnhance.Sharpness(pil_img).enhance(factor), dtype=np.uint8)

    def _add_hue_saturation(image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(_as_uint8(image), cv2.COLOR_RGB2HSV).astype(np.int16)
        hue_delta = int(np.random.randint(-25, 26))
        sat_delta = int(np.random.randint(-50, 51))
        hsv[..., 0] = np.mod(hsv[..., 0] + hue_delta, 180)
        hsv[..., 1] = np.clip(hsv[..., 1] + sat_delta, 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] + sat_delta, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    def _solarize(image: np.ndarray) -> np.ndarray:
        pil_img = Image.fromarray(_as_uint8(image))
        if np.random.rand() < 0.5:
            threshold = int(np.random.randint(32, 129))
            pil_img = ImageOps.solarize(pil_img, threshold=threshold)
        return np.asarray(pil_img, dtype=np.uint8)

    def _posterize(image: np.ndarray) -> np.ndarray:
        pil_img = Image.fromarray(_as_uint8(image))
        bits = int(np.random.randint(4, 9))
        return np.asarray(ImageOps.posterize(pil_img, bits=bits), dtype=np.uint8)

    def _invert(image: np.ndarray) -> np.ndarray:
        return 255 - _as_uint8(image)

    def _autocontrast(image: np.ndarray) -> np.ndarray:
        pil_img = Image.fromarray(_as_uint8(image))
        return np.asarray(ImageOps.autocontrast(pil_img), dtype=np.uint8)

    def _equalize(image: np.ndarray) -> np.ndarray:
        pil_img = Image.fromarray(_as_uint8(image))
        return np.asarray(ImageOps.equalize(pil_img), dtype=np.uint8)

    def _affine_rotate(image: np.ndarray) -> np.ndarray:
        image = _as_uint8(image)
        angle = float(np.random.uniform(-45.0, 45.0))
        h, w = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        return cv2.warpAffine(
            image,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    return [
        _CallableAugmenter(_gamma_contrast),
        _CallableAugmenter(_multiply_add_brightness),
        _CallableAugmenter(_enhance_sharpness),
        _CallableAugmenter(_add_hue_saturation),
        _CallableAugmenter(_solarize),
        _CallableAugmenter(_posterize),
        _CallableAugmenter(_invert),
        _CallableAugmenter(_autocontrast),
        _CallableAugmenter(_equalize),
        _CallableAugmenter(_affine_rotate),
    ]


class _SequentialAugmenter:
    """Minimal sequential augmenter compatible with the official call site."""

    def __init__(self, augmenters):
        self.augmenters = list(augmenters)

    def __call__(self, image=None, **kwargs):
        if image is None:
            image = kwargs['image']
        for augmenter in self.augmenters:
            image = augmenter(image=image)
        return image


def _lerp_np(x, y, w):
    return (y - x) * w + x


def rand_perlin_2d_np(
    shape: Tuple[int, int],
    res: Tuple[int, int],
    fade=lambda t: 6 * t**5 - 15 * t**4 + 10 * t**3,
):
    """Generate 2D Perlin noise."""
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)

    tile_grads = lambda s1, s2: np.repeat(
        np.repeat(gradients[s1[0]:s1[1], s2[0]:s2[1]], d[0], axis=0),
        d[1],
        axis=1,
    )
    dot = lambda grad, shift: (
        np.stack(
            (
                grid[:shape[0], :shape[1], 0] + shift[0],
                grid[:shape[0], :shape[1], 1] + shift[1],
            ),
            axis=-1,
        )
        * grad[:shape[0], :shape[1]]
    ).sum(axis=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * _lerp_np(
        _lerp_np(n00, n10, t[..., 0]),
        _lerp_np(n01, n11, t[..., 0]),
        t[..., 1],
    )


@DATASETS.register_module()
class RealNetTrainDataset(BaseDataset):
    """RealNet training dataset aligned to the official MVTec setup."""

    METAINFO: dict = dict(task='anomaly_detection')
    ALL_CATEGORIES: Sequence[str] = (
        'bottle', 'cable', 'capsule', 'carpet', 'grid',
        'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
        'tile', 'toothbrush', 'transistor', 'wood', 'zipper',
    )

    def __init__(
        self,
        data_root: str,
        cls_names: Optional[List[str]] = None,
        img_size: int | Tuple[int, int] = 256,
        dataset_type: str = 'mvtec',
        dtd_dir: Optional[str] = 'auto',
        sdas_dir: Optional[str] = 'auto',
        dtd_transparency_range: Tuple[float, float] = (0.2, 1.0),
        sdas_transparency_range: Tuple[float, float] = (0.5, 1.0),
        perlin_scale: int = 6,
        min_perlin_scale: int = 0,
        perlin_noise_threshold: float = 0.5,
        anomaly_types: Optional[Dict[str, float]] = None,
        pixel_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        pixel_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        pipeline: Optional[List[dict]] = None,
        multi_class: bool = False,
        **kwargs,
    ) -> None:
        del multi_class  # For benchmark compatibility, RealNet trains per-category.

        self._source_data_root = data_root
        self.cls_names = cls_names if cls_names else list(self.ALL_CATEGORIES)
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.resize = tuple(int(v) for v in img_size)
        self.dataset_type = dataset_type
        self.dtd_transparency_range = tuple(float(v) for v in dtd_transparency_range)
        self.sdas_transparency_range = tuple(float(v) for v in sdas_transparency_range)
        self.perlin_scale = perlin_scale
        self.min_perlin_scale = min_perlin_scale
        self.perlin_noise_threshold = perlin_noise_threshold
        self.anomaly_types = anomaly_types or {'normal': 0.5, 'sdas': 0.5}
        self.pixel_mean = np.asarray(pixel_mean, dtype=np.float32).reshape(3, 1, 1)
        self.pixel_std = np.asarray(pixel_std, dtype=np.float32).reshape(3, 1, 1)

        self._dtd_file_list = self._load_dtd_files(dtd_dir)
        self._sdas_files = self._load_sdas_files(sdas_dir)
        self._augmenters = _get_imgaug_augmenters()
        self._validate_anomaly_sources()

        pipeline = pipeline or []
        super().__init__(data_root=data_root, pipeline=pipeline, **kwargs)

    def _load_dtd_files(self, dtd_dir: Optional[str]) -> List[str]:
        if dtd_dir == 'auto':
            try:
                dtd_dir = _download_dtd()
            except Exception as exc:
                logger.warning('Failed to auto-download DTD dataset: %s', exc)
                dtd_dir = None

        if not dtd_dir:
            return []

        if os.path.isdir(os.path.join(dtd_dir, 'images')):
            dtd_dir = os.path.join(dtd_dir, 'images')

        files = sorted(
            glob.glob(os.path.join(dtd_dir, '*', '*.jpg'))
            + glob.glob(os.path.join(dtd_dir, '*', '*.png'))
            + glob.glob(os.path.join(dtd_dir, '*.jpg'))
            + glob.glob(os.path.join(dtd_dir, '*.png'))
        )
        if files:
            logger.info('RealNetTrainDataset: loaded %d DTD images', len(files))
        else:
            logger.warning('RealNetTrainDataset: no DTD images found at %s', dtd_dir)
        return files

    def _resolve_sdas_dir(self, sdas_dir: Optional[str], cls_name: str) -> Optional[str]:
        if sdas_dir in (None, 'auto'):
            base_dir = os.path.join(self._source_data_root, 'sdas')
        else:
            base_dir = sdas_dir

        if '{}' in base_dir:
            candidate = base_dir.format(cls_name)
            return candidate if os.path.isdir(candidate) else None

        if os.path.isdir(os.path.join(base_dir, cls_name)):
            return os.path.join(base_dir, cls_name)

        return base_dir if os.path.isdir(base_dir) else None

    def _load_sdas_files(self, sdas_dir: Optional[str]) -> Dict[str, List[str]]:
        file_map: Dict[str, List[str]] = {}
        for cls_name in self.cls_names:
            candidate_dir = self._resolve_sdas_dir(sdas_dir, cls_name)
            if not candidate_dir:
                continue
            files = sorted(
                glob.glob(os.path.join(candidate_dir, '*.png'))
                + glob.glob(os.path.join(candidate_dir, '*.jpg'))
                + glob.glob(os.path.join(candidate_dir, '*.jpeg'))
                + glob.glob(os.path.join(candidate_dir, '*.bmp'))
            )
            if files:
                file_map[cls_name] = files
        if file_map:
            logger.info(
                'RealNetTrainDataset: loaded SDAS files for %d categories',
                len(file_map),
            )
        return file_map

    def _validate_anomaly_sources(self) -> None:
        if self.anomaly_types.get('dtd', 0.0) > 0 and not self._dtd_file_list:
            raise FileNotFoundError(
                'RealNetTrainDataset requires DTD images because anomaly_types includes '
                "`dtd`, but no DTD files were found."
            )

        if self.anomaly_types.get('sdas', 0.0) > 0:
            missing = [cls_name for cls_name in self.cls_names if not self._sdas_files.get(cls_name)]
            if missing:
                missing_text = ', '.join(missing)
                raise FileNotFoundError(
                    'RealNetTrainDataset requires SDAS images because anomaly_types includes '
                    f"`sdas`, but no SDAS files were found for: {missing_text}."
                )

    def load_data_list(self) -> List[Dict]:
        data_list: List[Dict] = []
        for cls_name in self.cls_names:
            cls_dir = os.path.join(self.data_root, cls_name, 'train', 'good')
            if not os.path.isdir(cls_dir):
                continue

            for img_name in sorted(os.listdir(cls_dir)):
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    continue
                data_list.append(
                    dict(
                        img_path=os.path.join(cls_dir, img_name),
                        gt_mask_path='',
                        gt_label=0,
                        cls_name=cls_name,
                        defect_type='good',
                    )
                )
        return data_list

    def choice_anomaly_type(self, cls_name: str) -> str:
        del cls_name
        keys = list(self.anomaly_types)
        probs = [self.anomaly_types[key] for key in keys]
        return str(np.random.choice(keys, p=probs))

    def _rand_augmenter(self):
        aug_idx = np.random.choice(np.arange(len(self._augmenters)), 3, replace=False)
        return _SequentialAugmenter([
            self._augmenters[int(aug_idx[0])],
            self._augmenters[int(aug_idx[1])],
            self._augmenters[int(aug_idx[2])],
        ])

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        chw = image.transpose(2, 0, 1).astype(np.float32)
        return (chw - self.pixel_mean) / self.pixel_std

    def _resize_rgb_pil(self, image: np.ndarray) -> np.ndarray:
        pil_img = Image.fromarray(image, mode='RGB')
        resized = pil_img.resize(self.resize[::-1], resample=Image.BILINEAR)
        return np.asarray(resized)

    def __getitem__(self, idx: int) -> Dict:
        if not self._fully_initialized:
            self.full_init()

        data_info = self.get_data_info(idx)
        image = cv2.imread(data_info['img_path'], cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to load image: {data_info['img_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Match the official TrainBaseTransform path: PIL/tv Resize before
        # synthetic anomaly generation, rather than OpenCV resize.
        image = self._resize_rgb_pil(image)

        clean_image = image.astype(np.float32) / 255.0
        anomaly_type = self.choice_anomaly_type(data_info['cls_name'])

        if anomaly_type == 'normal':
            augmented = clean_image.copy()
            anomaly_mask = np.zeros(self.resize, dtype=np.float32)
        else:
            augmented, anomaly_mask = self.generate_anomaly(
                clean_image,
                data_info['cls_name'],
                anomaly_type,
            )

        results = dict(
            img=self._normalize(augmented),
            gt_img=self._normalize(clean_image),
            anomaly_mask=anomaly_mask.astype(np.float32),
            gt_label=0,
            gt_mask=anomaly_mask.astype(np.float32),
            cls_name=data_info['cls_name'],
            img_path=data_info['img_path'],
            defect_type=data_info['defect_type'],
            anomaly_type=anomaly_type,
        )
        return self.pipeline(results)

    def generate_anomaly(
        self,
        image: np.ndarray,
        cls_name: str,
        anomaly_type: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        foreground_mask = self.generate_target_foreground_mask(image, cls_name)
        perlin_mask = self.generate_perlin_noise_mask()
        mask = (perlin_mask * foreground_mask).astype(np.float32)

        if mask.sum() == 0:
            return image.copy(), np.zeros_like(mask, dtype=np.float32)

        source = self.anomaly_source(image, mask, cls_name, anomaly_type)
        return source.astype(np.float32), mask.astype(np.float32)

    def generate_target_foreground_mask(self, image: np.ndarray, cls_name: str) -> np.ndarray:
        if self.dataset_type != 'mvtec':
            raise NotImplementedError(f'Unsupported dataset type: {self.dataset_type}')

        img_gray = cv2.cvtColor((image * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        if cls_name in {'carpet', 'leather', 'tile', 'wood', 'cable', 'transistor'}:
            return np.ones_like(img_gray, dtype=np.float32)

        if cls_name == 'pill':
            _, fg = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            fg = (fg > 0).astype(np.uint8)
        elif cls_name in {'hazelnut', 'metal_nut', 'toothbrush'}:
            _, fg = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
            fg = (fg > 0).astype(np.uint8)
        elif cls_name in {'bottle', 'capsule', 'grid', 'screw', 'zipper'}:
            _, bg = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            fg = 1 - (bg > 0).astype(np.uint8)
        else:
            raise NotImplementedError(f'Unsupported MVTec foreground category: {cls_name}')

        fg = morphology.closing(fg.astype(bool), morphology.square(6))
        fg = morphology.opening(fg, morphology.square(6))
        return fg.astype(np.float32)

    def generate_perlin_noise_mask(self) -> np.ndarray:
        perlin_scalex = 2 ** int(torch.randint(self.min_perlin_scale, self.perlin_scale, (1,)).item())
        perlin_scaley = 2 ** int(torch.randint(self.min_perlin_scale, self.perlin_scale, (1,)).item())
        perlin_noise = rand_perlin_2d_np(self.resize, (perlin_scalex, perlin_scaley))
        iaa = _import_imgaug()
        perlin_noise = iaa.Affine(rotate=(-90, 90))(image=perlin_noise)
        return np.where(
            perlin_noise > self.perlin_noise_threshold,
            np.ones_like(perlin_noise),
            np.zeros_like(perlin_noise),
        ).astype(np.float32)

    def anomaly_source(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        cls_name: str,
        anomaly_type: str,
    ) -> np.ndarray:
        if anomaly_type == 'sdas':
            source = self._sdas_source(cls_name)
            factor = float(np.random.uniform(*self.sdas_transparency_range))
        elif anomaly_type == 'dtd':
            source = self._dtd_source()
            factor = float(np.random.uniform(*self.dtd_transparency_range))
        else:
            raise NotImplementedError(f'Unknown anomaly type: {anomaly_type}')

        mask_expanded = np.expand_dims(mask, axis=2)
        anomaly_source = factor * (mask_expanded * source) + (1 - factor) * (mask_expanded * image)
        return ((-mask_expanded + 1) * image) + anomaly_source

    def _dtd_source(self) -> np.ndarray:
        if not self._dtd_file_list:
            return np.random.rand(self.resize[0], self.resize[1], 3).astype(np.float32)

        src_path = str(np.random.choice(self._dtd_file_list))
        source = cv2.imread(src_path, cv2.IMREAD_COLOR)
        if source is None:
            return np.random.rand(self.resize[0], self.resize[1], 3).astype(np.float32)
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        source = cv2.resize(source, self.resize[::-1], interpolation=cv2.INTER_LINEAR)
        source = self._rand_augmenter()(image=source)
        return source.astype(np.float32) / 255.0

    def _sdas_source(self, cls_name: str) -> np.ndarray:
        files = self._sdas_files.get(cls_name, [])
        if not files:
            return np.random.rand(self.resize[0], self.resize[1], 3).astype(np.float32)

        src_path = str(np.random.choice(files))
        source = cv2.imread(src_path, cv2.IMREAD_COLOR)
        if source is None:
            return np.random.rand(self.resize[0], self.resize[1], 3).astype(np.float32)
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        source = cv2.resize(source, self.resize[::-1], interpolation=cv2.INTER_LINEAR)
        return source.astype(np.float32) / 255.0

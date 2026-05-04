"""DRAEM-specific dataset with anomaly augmentation in __getitem__.

This implementation matches the reference DRAEM code where augmentation
happens in the Dataset, not in the model. This ensures each sample has
exactly one augmented version per epoch, providing consistent training signals.
"""
import math
import os
import glob
import tarfile
import logging
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
from mmengine.dataset import BaseDataset

from baoiad.registry import DATASETS
from baoiad.models.detectors.draem import generate_perlin_mask

logger = logging.getLogger(__name__)

from baoiad.utils.dtd import download_dtd as _download_dtd


def _aug_gamma_contrast(img, gamma_range=(0.5, 2.0), per_channel=True):
    gamma = np.random.uniform(*gamma_range)
    if per_channel:
        gammas = [np.random.uniform(*gamma_range) for _ in range(img.shape[2])]
        out = np.stack([
            np.clip(((img[:, :, c] / 255.0) ** gammas[c]) * 255, 0, 255).astype(np.uint8)
            for c in range(img.shape[2])
        ], axis=2)
    else:
        out = np.clip(((img / 255.0) ** gamma) * 255, 0, 255).astype(np.uint8)
    return out


def _aug_multiply(img, mul_range=(0.8, 1.2), per_channel=True):
    if per_channel:
        out = np.stack([
            np.clip(img[:, :, c] * np.random.uniform(*mul_range), 0, 255).astype(np.uint8)
            for c in range(img.shape[2])
        ], axis=2)
    else:
        out = np.clip(img * np.random.uniform(*mul_range), 0, 255).astype(np.uint8)
    return out


def _aug_add(img, add_range=(-30, 30), per_channel=True):
    if per_channel:
        out = np.stack([
            np.clip(img[:, :, c].astype(np.int16) + np.random.randint(add_range[0], add_range[1] + 1), 0, 255).astype(np.uint8)
            for c in range(img.shape[2])
        ], axis=2)
    else:
        out = np.clip(img.astype(np.int16) + np.random.randint(add_range[0], add_range[1] + 1), 0, 255).astype(np.uint8)
    return out


def _aug_sharpen(img):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
    out = cv2.filter2D(img, -1, kernel)
    return np.clip(out, 0, 255).astype(np.uint8)


def _aug_hue_saturation(img, hue_range=(-50, 50)):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int16)
    hsv[:, :, 0] = np.clip(hsv[:, :, 0] + np.random.randint(hue_range[0], hue_range[1] + 1), 0, 179)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + np.random.randint(hue_range[0], hue_range[1] + 1), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def _aug_solarize(img, threshold_range=(32, 128)):
    threshold = np.random.randint(*threshold_range)
    out = img.copy()
    mask = out > threshold
    out[mask] = 255 - out[mask]
    return out


def _aug_posterize(img, bits=4):
    shift = 8 - bits
    out = (img >> shift) << shift
    return out.astype(np.uint8)


def _aug_invert(img):
    return (255 - img).astype(np.uint8)


def _aug_autocontrast(img):
    out = np.zeros_like(img)
    for c in range(img.shape[2]):
        ch = img[:, :, c]
        lo, hi = ch.min(), ch.max()
        if hi > lo:
            out[:, :, c] = np.clip(((ch.astype(np.float32) - lo) / (hi - lo) * 255), 0, 255).astype(np.uint8)
        else:
            out[:, :, c] = ch
    return out


def _aug_equalize(img):
    out = np.zeros_like(img)
    for c in range(img.shape[2]):
        out[:, :, c] = cv2.equalizeHist(img[:, :, c])
    return out


def _aug_rotate(img, angle_range=(-45, 45)):
    angle = np.random.uniform(*angle_range)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT101)


def _get_texture_augmenters():
    """Get texture augmenters as pure cv2/numpy callables."""
    return [
        lambda x: _aug_gamma_contrast(x, (0.5, 2.0), per_channel=True),
        lambda x: _aug_multiply(x, (0.8, 1.2), per_channel=True),
        lambda x: _aug_add(x, (-30, 30), per_channel=True),
        _aug_sharpen,
        lambda x: _aug_hue_saturation(x, (-50, 50)),
        lambda x: _aug_solarize(x, (32, 128)),
        _aug_posterize,
        _aug_invert,
        _aug_autocontrast,
        _aug_equalize,
        lambda x: _aug_rotate(x, (-45, 45)),
    ]


@DATASETS.register_module(force=True)
class DRAEMDataset(BaseDataset):
    """DRAEM training dataset with augmentation in __getitem__.

    This matches the reference implementation exactly:
    1. Load image as uint8
    2. 30% probability: rotate original image with imgaug (-90, 90)
    3. Convert to float [0, 1]
    4. 50% probability: return clean image + zero mask
    5. Otherwise: generate Perlin noise + DTD texture anomaly

    Args:
        data_root: Root directory of MVTec AD dataset.
        cls_names: List of category names to include.
        dtd_path: Path to DTD texture dataset. 'auto' for auto-download.
        img_size: Target image size.
        beta_range: Range for random beta blending factor.
        anomaly_ratio: Probability of generating an anomaly (vs clean image).
        pipeline: Data processing pipeline (should include PackDRAEMInputs).
    """

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
        dtd_path: str = 'auto',
        img_size: int = 256,
        beta_range: tuple = (0.0, 0.8),
        anomaly_ratio: float = 0.5,
        pipeline: Optional[List[dict]] = None,
        multi_class: bool = False,  # Accept but ignore (for benchmark.py compatibility)
        **kwargs,
    ):
        self.cls_names = cls_names if cls_names else list(self.ALL_CATEGORIES)
        self.img_size = img_size
        self.beta_range = beta_range
        self.anomaly_ratio = anomaly_ratio
        # multi_class is ignored - DRAEM always trains per-category

        # Load DTD texture paths
        self.anomaly_source_paths = []
        effective_dtd_path = dtd_path

        if dtd_path == 'auto':
            try:
                effective_dtd_path = _download_dtd()
            except Exception as e:
                logger.warning(f"Failed to auto-download DTD dataset: {e}. Using random noise textures.")
                effective_dtd_path = None

        if effective_dtd_path and os.path.isdir(effective_dtd_path):
            self.anomaly_source_paths = sorted(
                glob.glob(os.path.join(effective_dtd_path, '*', '*.jpg')) +
                glob.glob(os.path.join(effective_dtd_path, '*', '*.png')) +
                glob.glob(os.path.join(effective_dtd_path, '*.jpg')) +
                glob.glob(os.path.join(effective_dtd_path, '*.png'))
            )
            if self.anomaly_source_paths:
                logger.info(f"DRAEMDataset: Loaded {len(self.anomaly_source_paths)} DTD texture images")
            else:
                logger.warning(f"DRAEMDataset: No texture images found at {effective_dtd_path}, using random noise")

        # Cache texture augmenters (pure cv2/numpy, no imgaug dependency)
        self._augmenters = _get_texture_augmenters()

        # Solarize threshold sampled once at init (matching anomalib)
        self._solarize_threshold = np.random.uniform(32.0 / 255.0, 128.0 / 255.0)

        pipeline = pipeline or []
        super().__init__(data_root=data_root, pipeline=pipeline, **kwargs)

    def load_data_list(self) -> List[Dict]:
        """Load training image paths from MVTec AD or VisA dataset."""
        data_list: List[Dict] = []
        for cls_name in self.cls_names:
            # Try MVTec path first, then VisA path
            cls_dir = os.path.join(self.data_root, cls_name, 'train', 'good')
            if not os.path.isdir(cls_dir):
                cls_dir = os.path.join(self.data_root, cls_name, 'Data', 'Images', 'Normal')
            if not os.path.isdir(cls_dir):
                continue

            for img_name in sorted(os.listdir(cls_dir)):
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                    continue

                img_path = os.path.join(cls_dir, img_name)
                data_list.append(dict(
                    img_path=img_path,
                    gt_label=0,  # All training images are normal
                    gt_mask_path='',
                    cls_name=cls_name,
                    defect_type='good',
                ))

        return data_list

    def __getitem__(self, idx: int) -> Dict:
        """Get item with augmentation.

        Returns dict with keys:
            - img: Original image tensor (C, H, W) in [0, 1]
            - augmented_img: Augmented image tensor (C, H, W) in [0, 1]
            - anomaly_mask: Binary mask tensor (H, W)
            - gt_label: Ground truth label (0 for training)
            - cls_name: Category name
            - img_path: Path to original image
            - defect_type: Defect type ('good' for training)
        """
        # Ensure dataset is fully initialized (like BaseDataset.__getitem__ does)
        if not self._fully_initialized:
            self.full_init()

        # Use get_data_info() to handle serialized data properly
        data_info = self.get_data_info(idx)

        # 1. Load image as uint8
        img_uint8 = cv2.imread(data_info['img_path'], cv2.IMREAD_COLOR)
        if img_uint8 is None:
            raise FileNotFoundError(f"Failed to load image: {data_info['img_path']}")
        img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2RGB)

        # 2. Resize to target size
        H, W = self.img_size, self.img_size
        img_uint8 = cv2.resize(img_uint8, (W, H))

        # 3. 30% probability: rotate original image (-90, 90)
        if np.random.random() > 0.7:
            img_uint8 = _aug_rotate(img_uint8, (-90, 90))

        # 4. Convert to float [0, 1] - this is the original image
        img_float = img_uint8.astype(np.float32) / 255.0

        # 5. Decide if generating anomaly
        if np.random.random() > self.anomaly_ratio:
            # No anomaly: return original image + zero mask
            augmented_img = img_float.copy()
            anomaly_mask = np.zeros((H, W), dtype=np.float32)
        else:
            # Generate anomaly
            augmented_img, anomaly_mask = self._generate_anomaly(img_float, H, W)

        # Build results dict
        results = dict(
            img=img_float.transpose(2, 0, 1),  # (C, H, W)
            augmented_img=augmented_img.transpose(2, 0, 1),  # (C, H, W)
            anomaly_mask=anomaly_mask,  # (H, W)
            gt_label=data_info['gt_label'],
            cls_name=data_info['cls_name'],
            img_path=data_info['img_path'],
            defect_type=data_info['defect_type'],
        )

        # Apply pipeline (e.g., PackDRAEMInputs)
        return self.pipeline(results)

    def _generate_anomaly(self, img_float: np.ndarray, H: int, W: int) -> tuple:
        """Generate synthetic anomaly for a float image.

        Args:
            img_float: Image in float [0, 1] range, shape (H, W, C)
            H: Image height
            W: Image width

        Returns:
            augmented_img: Augmented image (H, W, C) in [0, 1]
            anomaly_mask: Binary mask (H, W)
        """
        C = img_float.shape[2]

        # Generate Perlin noise mask
        perlin_mask = generate_perlin_mask(H, W)

        if perlin_mask.sum() == 0:
            return img_float.copy(), np.zeros((H, W), dtype=np.float32)

        # Get anomaly source texture
        if self.anomaly_source_paths:
            src_path = self.anomaly_source_paths[np.random.randint(len(self.anomaly_source_paths))]
            src_img = cv2.imread(src_path)
            if src_img is not None:
                src_img = cv2.resize(src_img, (W, H))
                src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
                texture = src_img.astype(np.float32) / 255.0

                # Apply random augmentations to texture (pick 3)
                aug_ind = np.random.choice(len(self._augmenters), 3, replace=False)
                texture_uint8 = (texture * 255).astype(np.uint8)
                for idx in aug_ind:
                    texture_uint8 = self._augmenters[idx](texture_uint8)
                texture = texture_uint8.astype(np.float32) / 255.0
            else:
                texture = np.random.rand(H, W, C).astype(np.float32)
        else:
            # Fallback: random noise texture
            texture = np.random.rand(H, W, C).astype(np.float32)

        # Beta blending matching ADer: beta in [beta_min, beta_max]
        # ADer formula: augmented = image * (1-mask) + (1-beta) * texture * mask + beta * image * mask
        beta_min, beta_max = self.beta_range
        beta = np.random.random() * (beta_max - beta_min) + beta_min

        mask_3d = perlin_mask[:, :, np.newaxis]  # (H, W, 1)
        augmented = img_float * (1 - mask_3d) + (1 - beta) * texture * mask_3d + beta * img_float * mask_3d

        return augmented.astype(np.float32), perlin_mask.astype(np.float32)

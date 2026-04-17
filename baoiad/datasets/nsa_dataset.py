"""NSA-specific training dataset.

This mirrors the upstream NSA training data path more closely than the generic
pipeline:
- keep worker-local previous-sample state (`prev_idx`)
- apply per-category train transforms before synthesis
- generate logistic-intensity labels in the dataset, not in the detector
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
from mmengine.dataset import BaseDataset
from PIL import Image
from torchvision import transforms as T
from baoiad.models.detectors.nsa import (
    CV2_MIXED_CLONE,
    CV2_NORMAL_CLONE,
    TEXTURE_CATEGORIES,
    _official_patch_ex,
    DEFAULT_RESIZE_BOUNDS,
    TEXTURE_RESIZE_BOUNDS,
    WIDTH_BOUNDS_PCT,
    DEFAULT_WIDTH_BOUNDS,
    INTENSITY_LOGISTIC_PARAMS,
    DEFAULT_LOGISTIC_PARAMS,
    NUM_PATCHES,
    DEFAULT_NUM_PATCHES,
    BACKGROUND_PARAMS,
    MIN_OBJECT_PCT,
    MIN_OVERLAP_PCT,
)
from baoiad.registry import DATASETS


@DATASETS.register_module()
class NSATrainDataset(BaseDataset):
    """NSA training dataset with upstream-style worker-local source selection."""

    METAINFO: dict = dict(task='anomaly_detection')
    ALL_CATEGORIES: Sequence[str] = (
        'bottle', 'cable', 'capsule', 'carpet', 'grid',
        'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
        'tile', 'toothbrush', 'transistor', 'wood', 'zipper',
    )

    UNALIGNED_OBJECTS = {'bottle', 'hazelnut', 'metal_nut', 'screw'}
    ALIGNED_OBJECTS = {'cable', 'capsule', 'pill', 'transistor', 'toothbrush', 'zipper'}

    def __init__(
        self,
        data_root: str,
        cls_names: Optional[List[str]] = None,
        anomaly_ratio: float = 1.0,
        use_logistic_labels: bool = True,
        pipeline: Optional[List[dict]] = None,
        multi_class: bool = False,
        **kwargs,
    ) -> None:
        del multi_class  # NSA trains per category, but keep signature benchmark-compatible.
        self.cls_names = cls_names if cls_names else list(self.ALL_CATEGORIES)
        self.anomaly_ratio = float(anomaly_ratio)
        self.use_logistic_labels = bool(use_logistic_labels)

        if not hasattr(Image, 'ANTIALIAS'):
            Image.ANTIALIAS = Image.Resampling.LANCZOS  # type: ignore[attr-defined]

        self._resize_256 = T.Resize(256, Image.ANTIALIAS)
        self._resize_264 = T.Resize(264, Image.ANTIALIAS)
        self._object_unaligned_transform = T.Compose([
            T.RandomRotation(5),
            T.CenterCrop(230),
            T.RandomCrop(224),
        ])
        self._object_aligned_transform = T.Compose([
            T.CenterCrop(230),
            T.RandomCrop(224),
        ])
        self._texture_transform = T.Compose([
            T.RandomVerticalFlip(),
            T.RandomCrop(256),
        ])

        self._indices_by_class: Dict[str, List[int]] = {}
        self._prev_index_by_class: Dict[str, Optional[int]] = {}
        self._image_cache: Dict[int, Image.Image] = {}

        super().__init__(data_root=data_root, pipeline=pipeline or [], **kwargs)

    def load_data_list(self) -> List[Dict]:
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
                data_list.append(
                    dict(
                        img_path=img_path,
                        gt_label=0,
                        gt_mask_path='',
                        cls_name=cls_name,
                        defect_type='good',
                    ))
        return data_list

    def _ensure_class_state(self) -> None:
        if self._indices_by_class:
            return
        for idx in range(len(self)):
            info = self.get_data_info(idx)
            self._indices_by_class.setdefault(info['cls_name'], []).append(idx)
        for cls_name in self._indices_by_class:
            self._prev_index_by_class.setdefault(
                cls_name,
                int(self._indices_by_class[cls_name][np.random.randint(len(self._indices_by_class[cls_name]))]),
            )

    @staticmethod
    def _load_rgb_image(path: str) -> Image.Image:
        if not os.path.exists(path):
            raise FileNotFoundError(f'Failed to load image: {path}')
        return Image.open(path).convert('RGB')

    @staticmethod
    def _to_uint8(image: np.ndarray) -> np.ndarray:
        if image.dtype == np.uint8:
            return image
        return np.clip(np.round(image), 0, 255).astype(np.uint8)

    def _get_base_image(self, idx: int, cls_name: str) -> Image.Image:
        cached = self._image_cache.get(idx)
        if cached is not None:
            return cached.copy()

        image = self._load_rgb_image(self.get_data_info(idx)['img_path'])
        if cls_name in TEXTURE_CATEGORIES:
            image = self._resize_264(image)
        else:
            image = self._resize_256(image)
        self._image_cache[idx] = image
        return image.copy()

    def _apply_train_transform(self, image: Image.Image, cls_name: str) -> np.ndarray:
        if cls_name in TEXTURE_CATEGORIES:
            transformed = self._texture_transform(image)
        else:
            if cls_name in self.UNALIGNED_OBJECTS:
                transformed = self._object_unaligned_transform(image)
            else:
                transformed = self._object_aligned_transform(image)

        return self._to_uint8(np.asarray(transformed))

    def _get_prev_index(self, cls_name: str) -> int:
        prev_idx = self._prev_index_by_class.get(cls_name)
        if prev_idx is None:
            indices = self._indices_by_class[cls_name]
            prev_idx = int(indices[np.random.randint(len(indices))])
            self._prev_index_by_class[cls_name] = prev_idx
        return int(prev_idx)

    def _build_raw_sample(self, idx: int, update_state: bool = True, include_debug: bool = False) -> Dict:
        if not self._fully_initialized:
            self.full_init()
        self._ensure_class_state()

        data_info = self.get_data_info(idx)
        cls_name = data_info['cls_name']

        prev_idx = self._get_prev_index(cls_name)
        prev_info = self.get_data_info(prev_idx)

        current_base = self._get_base_image(idx, cls_name)
        source_base = self._get_base_image(prev_idx, cls_name)
        current_img = self._apply_train_transform(current_base.copy(), cls_name)
        source_img = self._apply_train_transform(source_base.copy(), cls_name)

        if self.anomaly_ratio >= 1.0:
            has_anomaly = 1.0
        elif self.anomaly_ratio <= 0.0:
            has_anomaly = 0.0
        else:
            has_anomaly = float(np.random.random() < self.anomaly_ratio)
        if has_anomaly:
            patched, label = _official_patch_ex(
                ima_dest=current_img,
                ima_src=source_img,
                mode=CV2_MIXED_CLONE if cls_name in TEXTURE_CATEGORIES else CV2_NORMAL_CLONE,
                width_bounds_pct=WIDTH_BOUNDS_PCT.get(cls_name, DEFAULT_WIDTH_BOUNDS),
                min_object_pct=MIN_OBJECT_PCT.get(cls_name),
                min_overlap_pct=MIN_OVERLAP_PCT.get(cls_name),
                label_mode='logistic-intensity' if self.use_logistic_labels else 'intensity',
                skip_background=BACKGROUND_PARAMS.get(cls_name),
                intensity_logistic_params=INTENSITY_LOGISTIC_PARAMS.get(cls_name, DEFAULT_LOGISTIC_PARAMS),
                num_patches=NUM_PATCHES.get(cls_name, DEFAULT_NUM_PATCHES),
                resize_bounds=TEXTURE_RESIZE_BOUNDS if cls_name in TEXTURE_CATEGORIES else DEFAULT_RESIZE_BOUNDS,
            )
        else:
            patched = current_img.copy()
            label = np.zeros(current_img.shape[:2] + (1,), dtype=np.float32)

        if update_state:
            self._prev_index_by_class[cls_name] = int(idx)

        results = dict(
            img=patched.astype(np.float32) / 255.0,
            gt_mask=label[..., 0].astype(np.float32),
            gt_label=0,
            cls_name=cls_name,
            img_path=data_info['img_path'],
            defect_type=data_info['defect_type'],
            source_index=int(prev_idx),
            source_img_path=prev_info['img_path'],
            has_anomaly=float(has_anomaly),
        )
        if include_debug:
            results.update(
                current_base_img=current_base,
                source_base_img=source_base,
                current_transformed_img=current_img,
                source_transformed_img=source_img,
                patched_uint8=patched,
            )
        return results

    def __getitem__(self, idx: int) -> Dict:
        results = self._build_raw_sample(idx, update_state=True, include_debug=False)
        return self.pipeline(results)

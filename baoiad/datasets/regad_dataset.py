"""RegAD official-compatible datasets.

The official RegAD implementation uses:

- cross-category training on normal images from all classes except the target
- few-shot support/query pairs during training
- raw ``Resize(224) -> ToTensor()`` preprocessing without ImageNet normalization

This module keeps the same semantics while exposing MMEngine-compatible
datasets for probing and custom training scripts.
"""

from __future__ import annotations

import os
import os.path as osp
import random
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from mmengine.dataset import BaseDataset
from PIL import Image

from baoiad.registry import DATASETS

_RESAMPLING = getattr(Image, 'Resampling', Image)
_RGB_RESAMPLE = _RESAMPLING.LANCZOS
_MASK_RESAMPLE = _RESAMPLING.NEAREST


def _load_rgb_tensor(img_path: str, img_size: int) -> torch.Tensor:
    """Load one RGB image using the official RegAD preprocessing path."""
    img = Image.open(img_path).convert('RGB')
    img = img.resize((img_size, img_size), _RGB_RESAMPLE)
    array = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1)).contiguous()


def _load_mask_tensor(mask_path: str, img_size: int) -> torch.Tensor:
    """Load one binary mask resized with nearest-neighbor interpolation."""
    if not mask_path or not osp.exists(mask_path):
        return torch.zeros((img_size, img_size), dtype=torch.float32)

    mask = Image.open(mask_path)
    mask = mask.resize((img_size, img_size), _MASK_RESAMPLE)
    array = np.asarray(mask, dtype=np.float32)
    if array.ndim == 3:
        array = array[..., 0]
    return torch.from_numpy((array > 0).astype(np.float32)).contiguous()


@DATASETS.register_module()
class RegADTrainDataset(BaseDataset):
    """Official-compatible RegAD training dataset.

    Each item contains one query image and ``shot`` support images sampled from
    the same source class, where the source class is drawn from all classes
    except the target class.
    """

    METAINFO: dict = dict(task='anomaly_detection')
    ALL_CATEGORIES: Sequence[str] = (
        # MVTec AD (15)
        'bottle', 'cable', 'capsule', 'carpet', 'grid',
        'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
        'tile', 'toothbrush', 'transistor', 'wood', 'zipper',
        # VisA (12)
        'candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
        'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4',
        'pipe_fryum',
    )

    def __init__(
        self,
        data_root: str,
        target_cls: str,
        split: str = 'train',
        img_size: int = 224,
        shot: int = 4,
        pipeline: Optional[List[dict]] = None,
        multi_class: bool = True,  # kept for benchmark/config compatibility
        cls_names: Optional[List[str]] = None,  # ignored, for benchmark compat
        **kwargs,
    ) -> None:
        if split != 'train':
            raise ValueError(
                f'RegADTrainDataset only supports split="train", got split="{split}"'
            )
        if target_cls not in self.ALL_CATEGORIES:
            raise ValueError(
                f'target_cls must be one of {self.ALL_CATEGORIES}, got "{target_cls}"'
            )
        if shot < 1:
            raise ValueError(f'shot must be >= 1, got {shot}')

        self.target_cls = target_cls
        self.split = split
        self.img_size = img_size
        self.shot = shot
        self.multi_class = multi_class
        self._train_class_to_paths: dict[str, list[str]] = {}

        kwargs.setdefault('serialize_data', False)
        pipeline = pipeline or []
        super().__init__(data_root=data_root, pipeline=pipeline, **kwargs)

    def _collect_train_paths(self) -> dict[str, list[str]]:
        class_to_paths: dict[str, list[str]] = {}
        for cls_name in self.ALL_CATEGORIES:
            if cls_name == self.target_cls:
                continue
            good_dir = osp.join(self.data_root, cls_name, self.split, 'good')
            if not osp.isdir(good_dir):
                continue
            img_paths = [
                osp.join(good_dir, img_name)
                for img_name in sorted(os.listdir(good_dir))
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
            ]
            if len(img_paths) >= 2:
                class_to_paths[cls_name] = img_paths
        return class_to_paths

    def _build_pair_records(self) -> list[dict]:
        data_list: list[dict] = []
        for cls_name, img_paths in self._train_class_to_paths.items():
            shuffled_paths = list(img_paths)
            random.shuffle(shuffled_paths)
            for query_path in shuffled_paths:
                support_paths = []
                for _ in range(self.shot):
                    support_path = random.choice(img_paths)
                    while support_path == query_path and len(img_paths) > 1:
                        support_path = random.choice(img_paths)
                    support_paths.append(support_path)
                data_list.append(
                    dict(
                        img_path=query_path,
                        support_img_paths=support_paths,
                        gt_label=0,
                        cls_name=cls_name,
                        source_cls=cls_name,
                        target_cls=self.target_cls,
                        defect_type='good',
                    )
                )
        return data_list

    def load_data_list(self) -> List[Dict]:
        self._train_class_to_paths = self._collect_train_paths()
        return self._build_pair_records()

    def shuffle_dataset(self) -> None:
        """Resample support images for a fresh epoch."""
        if not self._fully_initialized:
            self.full_init()
        self.data_list = self._build_pair_records()

    def __getitem__(self, idx: int) -> Dict:
        if not self._fully_initialized:
            self.full_init()

        data_info = self.get_data_info(idx)
        query_img = _load_rgb_tensor(data_info['img_path'], self.img_size)
        support_imgs = torch.stack(
            [_load_rgb_tensor(path, self.img_size) for path in data_info['support_img_paths']],
            dim=0,
        )
        results = dict(
            img=query_img,
            support_imgs=support_imgs,
            gt_label=data_info['gt_label'],
            cls_name=data_info['cls_name'],
            source_cls=data_info['source_cls'],
            target_cls=data_info['target_cls'],
            img_path=data_info['img_path'],
            defect_type=data_info['defect_type'],
        )
        return self.pipeline(results)


@DATASETS.register_module()
class RegADTestDataset(BaseDataset):
    """Official-compatible RegAD test dataset.

    This dataset loads query images from the target category. It can also load
    ``split='train'`` for support-set discovery on the target class.
    """

    METAINFO: dict = dict(task='anomaly_detection')
    ALL_CATEGORIES: Sequence[str] = RegADTrainDataset.ALL_CATEGORIES

    def __init__(
        self,
        data_root: str,
        target_cls: str,
        split: str = 'test',
        img_size: int = 224,
        pipeline: Optional[List[dict]] = None,
        multi_class: bool = False,  # kept for benchmark/config compatibility
        cls_names: Optional[List[str]] = None,  # ignored, for benchmark compat
        **kwargs,
    ) -> None:
        if target_cls not in self.ALL_CATEGORIES:
            raise ValueError(
                f'target_cls must be one of {self.ALL_CATEGORIES}, got "{target_cls}"'
            )
        if split not in {'train', 'test'}:
            raise ValueError(f'split must be "train" or "test", got "{split}"')

        self.target_cls = target_cls
        self.split = split
        self.img_size = img_size
        self.multi_class = multi_class

        kwargs.setdefault('serialize_data', False)
        pipeline = pipeline or []
        super().__init__(data_root=data_root, pipeline=pipeline, **kwargs)

    def load_data_list(self) -> List[Dict]:
        data_list: list[dict] = []
        cls_dir = osp.join(self.data_root, self.target_cls, self.split)
        if not osp.isdir(cls_dir):
            return data_list

        gt_dir = osp.join(self.data_root, self.target_cls, 'ground_truth')
        for defect_type in sorted(os.listdir(cls_dir)):
            defect_dir = osp.join(cls_dir, defect_type)
            if not osp.isdir(defect_dir):
                continue

            is_normal = defect_type == 'good'
            gt_label = 0 if is_normal else 1
            for img_name in sorted(os.listdir(defect_dir)):
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                    continue
                img_path = osp.join(defect_dir, img_name)
                gt_mask_path = ''
                if self.split == 'test' and not is_normal:
                    stem = osp.splitext(img_name)[0]
                    # Try MVTec naming ({stem}_mask.png), then VisA naming ({stem}.png)
                    gt_mask_path = osp.join(gt_dir, defect_type, f'{stem}_mask.png')
                    if not osp.isfile(gt_mask_path):
                        gt_mask_path = osp.join(gt_dir, defect_type, f'{stem}.png')
                data_list.append(
                    dict(
                        img_path=img_path,
                        gt_label=gt_label,
                        gt_mask_path=gt_mask_path,
                        cls_name=self.target_cls,
                        source_cls=self.target_cls,
                        target_cls=self.target_cls,
                        defect_type=defect_type,
                    )
                )
        return data_list

    def __getitem__(self, idx: int) -> Dict:
        if not self._fully_initialized:
            self.full_init()

        data_info = self.get_data_info(idx)
        img = _load_rgb_tensor(data_info['img_path'], self.img_size)
        gt_mask = _load_mask_tensor(data_info.get('gt_mask_path', ''), self.img_size)
        results = dict(
            img=img,
            gt_mask=gt_mask,
            gt_label=data_info['gt_label'],
            cls_name=data_info['cls_name'],
            source_cls=data_info['source_cls'],
            target_cls=data_info['target_cls'],
            img_path=data_info['img_path'],
            defect_type=data_info['defect_type'],
        )
        return self.pipeline(results)

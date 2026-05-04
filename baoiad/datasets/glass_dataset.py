"""Official-style GLASS dataset for strict alignment.

The strict GLASS training path performs augmentation inside the dataset and
expects additional assets such as DTD textures, foreground masks, and the
distribution metadata sheet used by the official codebase.
"""

from __future__ import annotations

import os
import os.path as osp
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from baoiad.datasets.base_ad_dataset import BaseADDataset
from baoiad.registry import DATASETS
from baoiad.utils.glass_utils import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    generate_glass_perlin_masks,
    resolve_dtd_texture_paths,
)


def _build_rand_augmenter(resize_arg, img_size: int):
    """Build the official 3-of-9 texture augmenter."""
    list_aug = [
        transforms.ColorJitter(contrast=(0.8, 1.2)),
        transforms.ColorJitter(brightness=(0.8, 1.2)),
        transforms.ColorJitter(saturation=(0.8, 1.2), hue=(-0.2, 0.2)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.RandomGrayscale(p=1.0),
        transforms.RandomAutocontrast(p=1.0),
        transforms.RandomEqualize(p=1.0),
        transforms.RandomAffine(degrees=(-45, 45)),
    ]
    aug_idx = np.random.choice(np.arange(len(list_aug)), 3, replace=False)
    return transforms.Compose([
        transforms.Resize(resize_arg),
        list_aug[int(aug_idx[0])],
        list_aug[int(aug_idx[1])],
        list_aug[int(aug_idx[2])],
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


@DATASETS.register_module(force=True)
class GLASSDataset(BaseADDataset):
    """Official-style GLASS dataset supporting both train and test splits."""

    ALL_CATEGORIES: Sequence[str] = (
        'bottle', 'cable', 'capsule', 'carpet', 'grid',
        'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
        'tile', 'toothbrush', 'transistor', 'wood', 'zipper',
    )

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        cls_names: Optional[List[str]] = None,
        multi_class: bool = False,
        dtd_path: Optional[str] = 'auto',
        dataset_name: str = 'mvtec',
        img_size: int = 288,
        resize: int = 288,
        rotate_degrees: float = 0.0,
        translate: float = 0.0,
        brightness_factor: float = 0.0,
        contrast_factor: float = 0.0,
        saturation_factor: float = 0.0,
        gray_p: float = 0.0,
        h_flip_p: float = 0.0,
        v_flip_p: float = 0.0,
        scale: float = 0.0,
        distribution: int = 0,
        mean: float = 0.5,
        std: float = 0.1,
        fg: int = 1,
        rand_aug: int = 1,
        downsampling: int = 8,
        fg_mask_root: Optional[str] = None,
        distribution_meta_path: Optional[str] = None,
        strict_assets_required: bool = True,
        max_mask_tries: int = 64,
        pipeline: Optional[List[dict]] = None,
        **kwargs,
    ) -> None:
        self.split = split
        self.dataset_name = dataset_name
        self.img_size = int(img_size)
        self.base_resize = int(resize)
        self.distribution = int(distribution)
        self.beta_mean = float(mean)
        self.beta_std = float(std)
        self.fg = int(fg)
        self.rand_aug = int(rand_aug)
        self.downsampling = int(downsampling)
        self.strict_assets_required = bool(strict_assets_required)
        self.distribution_meta_path = distribution_meta_path
        self.max_mask_tries = int(max_mask_tries)

        if self.downsampling <= 0 or self.img_size % self.downsampling != 0:
            raise ValueError(
                f'GLASSDataset requires img_size divisible by downsampling, got {self.img_size} and {self.downsampling}.'
            )

        self.texture_paths: list[str] = []
        if split == 'train':
            self.texture_paths = resolve_dtd_texture_paths(dtd_path)

        self.fg_mask_root = fg_mask_root
        if split == 'train' and self.fg != 0:
            if not self.fg_mask_root:
                self.fg_mask_root = osp.join(data_root, 'fg_mask')
            if self.strict_assets_required and not osp.isdir(self.fg_mask_root):
                raise FileNotFoundError(
                    f'GLASS strict training requires foreground masks at {self.fg_mask_root}.'
                )

        if split == 'train' and (self.distribution in {0, 4} or self.fg == 2):
            if not self.distribution_meta_path:
                raise FileNotFoundError(
                    'GLASS strict training requires distribution_meta_path when '
                    'distribution is file-driven or fg=2.'
                )
            if self.strict_assets_required and not osp.isfile(self.distribution_meta_path):
                raise FileNotFoundError(
                    f'GLASS strict training requires distribution metadata at {self.distribution_meta_path}.'
                )

        self._distribution_meta = None
        if self.distribution_meta_path and osp.isfile(self.distribution_meta_path):
            self._distribution_meta = pd.read_excel(self.distribution_meta_path)

        self._global_resize_arg = (self.base_resize, self.base_resize) if self.distribution == 1 else self.base_resize
        self._base_transform = transforms.Compose([
            transforms.Resize(self._global_resize_arg),
            transforms.ColorJitter(brightness_factor, contrast_factor, saturation_factor),
            transforms.RandomHorizontalFlip(h_flip_p),
            transforms.RandomVerticalFlip(v_flip_p),
            transforms.RandomGrayscale(gray_p),
            transforms.RandomAffine(
                rotate_degrees,
                translate=(translate, translate),
                scale=(1.0 - scale, 1.0 + scale),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        self._mask_transform = transforms.Compose([
            transforms.Resize(self._global_resize_arg),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
        ])

        super().__init__(
            data_root=data_root,
            split=split,
            cls_names=cls_names,
            multi_class=multi_class,
            pipeline=pipeline,
            **kwargs,
        )

    def load_data_list(self) -> List[Dict]:
        """Load the relevant MVTec image list."""
        data_list: List[Dict] = []
        for cls_name in self.cls_names:
            if self.split == 'train':
                train_dir = osp.join(self.data_root, cls_name, 'train', 'good')
                if not osp.isdir(train_dir):
                    continue
                for img_name in sorted(os.listdir(train_dir)):
                    if not img_name.lower().endswith(('.png', '.jpg', '.bmp')):
                        continue
                    data_list.append(dict(
                        img_path=osp.join(train_dir, img_name),
                        gt_label=0,
                        gt_mask_path='',
                        cls_name=cls_name,
                        defect_type='good',
                    ))
                continue

            cls_dir = osp.join(self.data_root, cls_name, 'test')
            if not osp.isdir(cls_dir):
                continue
            gt_dir = osp.join(self.data_root, cls_name, 'ground_truth')
            for defect_type in sorted(os.listdir(cls_dir)):
                defect_dir = osp.join(cls_dir, defect_type)
                if not osp.isdir(defect_dir):
                    continue
                is_normal = defect_type == 'good'
                gt_label = 0 if is_normal else 1
                for img_name in sorted(os.listdir(defect_dir)):
                    if not img_name.lower().endswith(('.png', '.jpg', '.bmp')):
                        continue
                    gt_mask_path = ''
                    if not is_normal:
                        stem = osp.splitext(img_name)[0]
                        candidate = osp.join(gt_dir, defect_type, f'{stem}_mask.png')
                        if osp.exists(candidate):
                            gt_mask_path = candidate
                    data_list.append(dict(
                        img_path=osp.join(defect_dir, img_name),
                        gt_label=gt_label,
                        gt_mask_path=gt_mask_path,
                        cls_name=cls_name,
                        defect_type=defect_type,
                    ))
        return data_list

    def _resize_arg_for_class(self, cls_name: str):
        if self.distribution == 1:
            return (self.base_resize, self.base_resize)
        if cls_name in {'toothbrush', 'wood'}:
            return round(self.img_size * 329 / 288)
        return self.base_resize

    def _image_transform_for_class(self, cls_name: str):
        resize_arg = self._resize_arg_for_class(cls_name)
        if resize_arg == self._global_resize_arg:
            return self._base_transform
        return transforms.Compose([
            transforms.Resize(resize_arg),
            *self._base_transform.transforms[1:],
        ])

    def _mask_transform_for_class(self, cls_name: str):
        resize_arg = self._resize_arg_for_class(cls_name)
        if resize_arg == self._global_resize_arg:
            return self._mask_transform
        return transforms.Compose([
            transforms.Resize(resize_arg),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
        ])

    def _distribution_row(self, cls_name: str) -> pd.Series:
        if self._distribution_meta is None:
            raise FileNotFoundError(
                'GLASS distribution metadata is required for the selected training mode.'
            )
        class_key = f'{self.dataset_name}_{cls_name}'
        row = self._distribution_meta.loc[self._distribution_meta['Class'] == class_key]
        if row.empty:
            raise KeyError(
                f'GLASS distribution metadata has no row for {class_key!r} in {self.distribution_meta_path}.'
            )
        return row.iloc[0]

    def foreground_policy(self, cls_name: str) -> int:
        """Return whether the current category should use foreground masks."""
        if self.fg == 0:
            return 0
        if self._distribution_meta is not None:
            row = self._distribution_row(cls_name)
            return int(row['Foreground'])
        if self.fg == 1:
            return 1
        return 1

    def _load_foreground_mask(self, data_info: Dict, cls_name: str) -> torch.Tensor:
        if self.foreground_policy(cls_name) == 0:
            return torch.ones(self.img_size, self.img_size, dtype=torch.float32)

        if not self.fg_mask_root:
            raise FileNotFoundError('GLASS foreground masks are enabled but fg_mask_root is not set.')

        fg_path = osp.join(self.fg_mask_root, cls_name, osp.basename(data_info['img_path']))
        if not osp.isfile(fg_path):
            raise FileNotFoundError(f'GLASS foreground mask missing: {fg_path}')

        transform_mask = self._mask_transform_for_class(cls_name)
        fg_mask = Image.open(fg_path).convert('L')
        fg_mask = torch.ceil(transform_mask(fg_mask)[0]).float()
        return fg_mask

    def _sample_texture(self, cls_name: str) -> torch.Tensor:
        if not self.texture_paths:
            raise FileNotFoundError('GLASS strict training requires non-empty DTD textures.')
        idx = int(torch.randint(0, len(self.texture_paths), (1,)).item())
        transform_aug = _build_rand_augmenter(self._resize_arg_for_class(cls_name), self.img_size)
        aug = Image.open(self.texture_paths[idx]).convert('RGB')
        return transform_aug(aug)

    def _build_training_sample(self, data_info: Dict) -> Dict:
        cls_name = data_info['cls_name']
        transform_img = self._image_transform_for_class(cls_name)
        image = Image.open(data_info['img_path']).convert('RGB')
        image = transform_img(image)

        aug = self._sample_texture(cls_name) if self.rand_aug else transform_img(
            Image.open(self.texture_paths[int(torch.randint(0, len(self.texture_paths), (1,)).item())]).convert('RGB')
        )
        mask_fg = self._load_foreground_mask(data_info, cls_name)
        feat_size = self.img_size // self.downsampling
        mask_s, mask_l = generate_glass_perlin_masks(
            img_shape=tuple(image.shape),
            feat_size=feat_size,
            min_scale=0,
            max_scale=6,
            mask_fg=mask_fg,
            return_large_mask=True,
            max_tries=self.max_mask_tries,
        )

        beta = np.random.normal(loc=self.beta_mean, scale=self.beta_std)
        beta = float(np.clip(beta, 0.2, 0.8))
        mask_l_t = torch.from_numpy(mask_l).float()
        aug_image = image * (1.0 - mask_l_t) + (1.0 - beta) * aug * mask_l_t + beta * image * mask_l_t

        return dict(
            img=image,
            aug=aug_image,
            mask_s=torch.from_numpy(mask_s).float(),
            gt_label=0,
            cls_name=cls_name,
            img_path=data_info['img_path'],
            defect_type=data_info['defect_type'],
        )

    def _build_test_sample(self, data_info: Dict) -> Dict:
        cls_name = data_info['cls_name']
        transform_img = self._image_transform_for_class(cls_name)
        transform_mask = self._mask_transform_for_class(cls_name)

        image = Image.open(data_info['img_path']).convert('RGB')
        image = transform_img(image)

        if data_info['gt_mask_path']:
            mask = Image.open(data_info['gt_mask_path']).convert('L')
            gt_mask = transform_mask(mask)[0].float()
        else:
            gt_mask = torch.zeros(self.img_size, self.img_size, dtype=torch.float32)

        return dict(
            img=image,
            gt_mask=gt_mask,
            gt_label=int(data_info['gt_label']),
            cls_name=cls_name,
            img_path=data_info['img_path'],
            defect_type=data_info['defect_type'],
        )

    def __getitem__(self, idx: int) -> Dict:
        if not self._fully_initialized:
            self.full_init()

        data_info = self.get_data_info(idx)
        if self.split == 'train':
            results = self._build_training_sample(data_info)
        else:
            results = self._build_test_sample(data_info)
        return self.pipeline(results)

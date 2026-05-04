"""MVTec AD dataset."""

import os
import os.path as osp
import random
from typing import Dict, List

from sklearn.model_selection import train_test_split

from baoiad.datasets.base_ad_dataset import BaseADDataset
from baoiad.registry import DATASETS


@DATASETS.register_module(force=True)
class MVTecADDataset(BaseADDataset):
    """MVTec Anomaly Detection dataset.

    Expected directory structure::

        data_root/
        ├── bottle/
        │   ├── train/
        │   │   └── good/
        │   │       ├── 000.png
        │   │       └── ...
        │   ├── test/
        │   │   ├── good/
        │   │   │   └── ...
        │   │   ├── broken_large/
        │   │   │   └── ...
        │   │   └── ...
        │   └── ground_truth/
        │       ├── broken_large/
        │       │   ├── 000_mask.png
        │       │   └── ...
        │       └── ...
        ├── cable/
        └── ...

    Args:
        data_root: Path to MVTec AD root directory.
        split: 'train' or 'test'.
        cls_names: Categories to include. None for all.
        multi_class: Whether to load multiple categories.
        shuffle_train_data: Whether to shuffle the train data list once during
            dataset construction. ADer's DefaultAD applies this before the
            DataLoader-level shuffle, so strict MUAD reproductions may opt in.
        train_val_split_ratio: Optional validation split ratio applied to the
            clean train set. Only used when ``split='train'``.
        train_val_split_seed: Random seed for the train/val split.
        train_val_split_subset: Which subset to expose when
            ``train_val_split_ratio > 0``. Supported values are ``'train'``,
            ``'val'`` or ``None``.
        pipeline: Data transform pipeline.
    """

    ALL_CATEGORIES = (
        'bottle', 'cable', 'capsule', 'carpet', 'grid',
        'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
        'tile', 'toothbrush', 'transistor', 'wood', 'zipper',
    )

    def __init__(
        self,
        *args,
        shuffle_train_data: bool = False,
        train_val_split_ratio: float = 0.0,
        train_val_split_seed: int = 0,
        train_val_split_subset: str | None = None,
        **kwargs,
    ):
        self.shuffle_train_data = shuffle_train_data
        self.train_val_split_ratio = float(train_val_split_ratio)
        self.train_val_split_seed = int(train_val_split_seed)
        self.train_val_split_subset = train_val_split_subset
        if self.train_val_split_subset not in {None, 'train', 'val'}:
            raise ValueError(
                'train_val_split_subset must be one of None, '
                f"'train', or 'val', got {self.train_val_split_subset!r}."
            )
        super().__init__(*args, **kwargs)

    def _split_train_subset(self, data_list: List[Dict]) -> List[Dict]:
        if self.split != 'train':
            return data_list
        if self.train_val_split_ratio <= 0 or self.train_val_split_subset is None:
            return data_list
        if len(data_list) <= 1:
            return data_list if self.train_val_split_subset == 'train' else []

        train_items, val_items = train_test_split(
            data_list,
            test_size=self.train_val_split_ratio,
            random_state=self.train_val_split_seed,
            shuffle=True,
        )
        return train_items if self.train_val_split_subset == 'train' else val_items

    def load_data_list(self) -> List[Dict]:
        """Load data annotations for MVTec AD.

        Returns:
            List of dicts with keys: img_path, gt_label, gt_mask_path,
            cls_name, defect_type.
        """
        data_list: List[Dict] = []
        for cls_name in self.cls_names:
            class_data_list: List[Dict] = []
            cls_dir = osp.join(self.data_root, cls_name, self.split)
            if not osp.isdir(cls_dir):
                continue

            gt_dir = osp.join(self.data_root, cls_name, 'ground_truth')

            for defect_type in sorted(os.listdir(cls_dir)):
                defect_dir = osp.join(cls_dir, defect_type)
                if not osp.isdir(defect_dir):
                    continue

                is_normal = (defect_type == 'good')
                gt_label = 0 if is_normal else 1

                for img_name in sorted(os.listdir(defect_dir)):
                    if not img_name.lower().endswith(('.png', '.jpg', '.bmp')):
                        continue

                    img_path = osp.join(defect_dir, img_name)

                    # Mask path for defective test images
                    gt_mask_path = ''
                    if self.split == 'test' and not is_normal:
                        stem = osp.splitext(img_name)[0]
                        # Try MVTec convention first: {stem}_mask.png
                        mask_name = f'{stem}_mask.png'
                        mask_path = osp.join(gt_dir, defect_type, mask_name)
                        if osp.exists(mask_path):
                            gt_mask_path = mask_path
                        else:
                            # Fallback: {stem}.png (VisA, BTech, etc.)
                            mask_name = f'{stem}.png'
                            mask_path = osp.join(gt_dir, defect_type, mask_name)
                            if osp.exists(mask_path):
                                gt_mask_path = mask_path

                    class_data_list.append(dict(
                        img_path=img_path,
                        gt_label=gt_label,
                        gt_mask_path=gt_mask_path,
                        cls_name=cls_name,
                        defect_type=defect_type,
                    ))

            data_list.extend(self._split_train_subset(class_data_list))

        if self.split == 'train' and self.shuffle_train_data:
            random.shuffle(data_list)

        return data_list

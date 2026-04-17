"""MVTec AD 2 dataset."""

import os
import os.path as osp
from typing import Dict, List, Optional

from baoiad.datasets.base_ad_dataset import BaseADDataset
from baoiad.registry import DATASETS


class TestType:
    """Type of test set for MVTec AD 2."""
    PUBLIC = 'public'
    PRIVATE = 'private'
    PRIVATE_MIXED = 'private_mixed'


@DATASETS.register_module()
class MVTecAD2Dataset(BaseADDataset):
    """MVTec AD 2 dataset for advanced anomaly detection scenarios.

    Expected directory structure::

        data_root/
        ├── can/
        │   ├── train/
        │   │   └── good/
        │   ├── validation/
        │   │   └── good/
        │   ├── test_public/
        │   │   ├── good/
        │   │   ├── bad/
        │   │   └── ground_truth/
        │   │       └── bad/
        │   ├── test_private/
        │   └── test_private_mixed/
        ├── fabric/
        └── ...

    Reference:
        L. Heckler-Kram et al., "The MVTec AD 2 Dataset: Advanced Scenarios
        for Unsupervised Anomaly Detection", arXiv 2024.

    Args:
        data_root: Path to MVTec AD 2 root directory.
        split: 'train', 'val', or 'test'.
        cls_names: Categories to include. None for all.
        multi_class: Whether to load multiple categories.
        pipeline: Data transform pipeline.
        test_type: Type of test set ('public', 'private', 'private_mixed').
            Only used when split='test'.
    """

    ALL_CATEGORIES = (
        'can',
        'fabric',
        'fruit_jelly',
        'rice',
        'sheet_metal',
        'vial',
        'wallplugs',
        'walnuts',
    )

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        cls_names: Optional[List[str]] = None,
        multi_class: bool = True,
        pipeline: Optional[List[dict]] = None,
        test_type: str = TestType.PUBLIC,
        **kwargs,
    ) -> None:
        self.test_type = test_type
        # Map 'val' to 'validation' for internal use
        self._split_dir = 'validation' if split == 'val' else split
        super().__init__(
            data_root=data_root,
            split=split,
            cls_names=cls_names,
            multi_class=multi_class,
            pipeline=pipeline,
            **kwargs,
        )

    def load_data_list(self) -> List[Dict]:
        """Load data annotations for MVTec AD 2.

        Returns:
            List of dicts with keys: img_path, gt_label, gt_mask_path,
            cls_name, defect_type.
        """
        data_list: List[Dict] = []

        for cls_name in self.cls_names:
            if self.split == 'train':
                data_list.extend(self._load_train_samples(cls_name))
            elif self.split == 'val':
                data_list.extend(self._load_val_samples(cls_name))
            elif self.split == 'test':
                data_list.extend(self._load_test_samples(cls_name))

        return data_list

    def _load_train_samples(self, cls_name: str) -> List[Dict]:
        """Load training samples (normal only)."""
        data_list: List[Dict] = []
        train_dir = osp.join(self.data_root, cls_name, 'train', 'good')
        if not osp.isdir(train_dir):
            return data_list

        for img_name in sorted(os.listdir(train_dir)):
            if not img_name.lower().endswith(('.png', '.jpg', '.bmp')):
                continue

            img_path = osp.join(train_dir, img_name)
            data_list.append(dict(
                img_path=img_path,
                gt_label=0,
                gt_mask_path='',
                cls_name=cls_name,
                defect_type='good',
            ))

        return data_list

    def _load_val_samples(self, cls_name: str) -> List[Dict]:
        """Load validation samples (normal only)."""
        data_list: List[Dict] = []
        val_dir = osp.join(self.data_root, cls_name, 'validation', 'good')
        if not osp.isdir(val_dir):
            return data_list

        for img_name in sorted(os.listdir(val_dir)):
            if not img_name.lower().endswith(('.png', '.jpg', '.bmp')):
                continue

            img_path = osp.join(val_dir, img_name)
            data_list.append(dict(
                img_path=img_path,
                gt_label=0,
                gt_mask_path='',
                cls_name=cls_name,
                defect_type='good',
            ))

        return data_list

    def _load_test_samples(self, cls_name: str) -> List[Dict]:
        """Load test samples based on test_type."""
        if self.test_type == TestType.PUBLIC:
            return self._load_public_test_samples(cls_name)
        elif self.test_type == TestType.PRIVATE:
            return self._load_private_test_samples(cls_name)
        elif self.test_type == TestType.PRIVATE_MIXED:
            return self._load_private_mixed_test_samples(cls_name)
        return []

    def _load_public_test_samples(self, cls_name: str) -> List[Dict]:
        """Load public test samples with ground truth."""
        data_list: List[Dict] = []
        test_dir = osp.join(self.data_root, cls_name, 'test_public')
        gt_dir = osp.join(test_dir, 'ground_truth')

        # Load normal test samples
        good_dir = osp.join(test_dir, 'good')
        if osp.isdir(good_dir):
            for img_name in sorted(os.listdir(good_dir)):
                if not img_name.lower().endswith(('.png', '.jpg', '.bmp')):
                    continue
                img_path = osp.join(good_dir, img_name)
                data_list.append(dict(
                    img_path=img_path,
                    gt_label=0,
                    gt_mask_path='',
                    cls_name=cls_name,
                    defect_type='good',
                ))

        # Load abnormal test samples
        bad_dir = osp.join(test_dir, 'bad')
        if osp.isdir(bad_dir):
            for img_name in sorted(os.listdir(bad_dir)):
                if not img_name.lower().endswith(('.png', '.jpg', '.bmp')):
                    continue

                img_path = osp.join(bad_dir, img_name)
                stem = osp.splitext(img_name)[0]
                mask_name = f'{stem}_mask.png'
                mask_path = osp.join(gt_dir, 'bad', mask_name)

                gt_mask_path = mask_path if osp.exists(mask_path) else ''

                data_list.append(dict(
                    img_path=img_path,
                    gt_label=1,
                    gt_mask_path=gt_mask_path,
                    cls_name=cls_name,
                    defect_type='bad',
                ))

        return data_list

    def _load_private_test_samples(self, cls_name: str) -> List[Dict]:
        """Load private test samples (no ground truth)."""
        data_list: List[Dict] = []
        test_dir = osp.join(self.data_root, cls_name, 'test_private')
        if not osp.isdir(test_dir):
            return data_list

        for img_name in sorted(os.listdir(test_dir)):
            if not img_name.lower().endswith(('.png', '.jpg', '.bmp')):
                continue

            img_path = osp.join(test_dir, img_name)
            # Private samples have unknown labels (use -1 or default to 0)
            data_list.append(dict(
                img_path=img_path,
                gt_label=-1,  # Unknown label for private test
                gt_mask_path='',
                cls_name=cls_name,
                defect_type='unknown',
            ))

        return data_list

    def _load_private_mixed_test_samples(self, cls_name: str) -> List[Dict]:
        """Load private mixed test samples (no ground truth)."""
        data_list: List[Dict] = []
        test_dir = osp.join(self.data_root, cls_name, 'test_private_mixed')
        if not osp.isdir(test_dir):
            return data_list

        for img_name in sorted(os.listdir(test_dir)):
            if not img_name.lower().endswith(('.png', '.jpg', '.bmp')):
                continue

            img_path = osp.join(test_dir, img_name)
            data_list.append(dict(
                img_path=img_path,
                gt_label=-1,  # Unknown label for private test
                gt_mask_path='',
                cls_name=cls_name,
                defect_type='unknown',
            ))

        return data_list

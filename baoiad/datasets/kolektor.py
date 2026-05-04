"""Kolektor Surface-Defect dataset."""

import os
import os.path as osp
from typing import Dict, List, Optional

import numpy as np

from baoiad.datasets.base_ad_dataset import BaseADDataset
from baoiad.registry import DATASETS


def _is_mask_anomalous(mask_path: str) -> bool:
    """Check if a mask shows defects.

    Args:
        mask_path: Path to the mask file.

    Returns:
        True if mask contains defects (non-zero pixels).
    """
    try:
        import cv2
        img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False
        return not np.all(img == 0)
    except ImportError:
        # Fallback using PIL if cv2 not available
        from PIL import Image
        img = np.array(Image.open(mask_path).convert('L'))
        return not np.all(img == 0)


@DATASETS.register_module(force=True)
class KolektorDataset(BaseADDataset):
    """Kolektor Surface-Defect dataset.

    Expected directory structure::

        data_root/
        ├── kos01/
        │   ├── Part0.jpg
        │   ├── Part0_label.bmp
        │   ├── Part1.jpg
        │   └── Part1_label.bmp
        ├── kos02/
        └── ...

    The dataset uses items (kos01, kos02, etc.) containing PartX.jpg images
    and corresponding PartX_label.bmp masks. Normal samples have all-zero masks.
    A random 80/20 split is used for good samples (seed=42), all bad samples
    go to test set.

    Reference:
        D. Tabernik et al., "Segmentation-based deep-learning approach for
        surface-defect detection", Journal of Intelligent Manufacturing 2020.

    Args:
        data_root: Path to Kolektor root directory.
        split: 'train' or 'test'.
        cls_names: Categories to include. None for all (uses 'kolektor').
        multi_class: Whether to load multiple categories.
        pipeline: Data transform pipeline.
        train_split_ratio: Ratio for splitting good images (default 0.8).
    """

    ALL_CATEGORIES = ('kolektor',)

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        cls_names: Optional[List[str]] = None,
        multi_class: bool = True,
        pipeline: Optional[List[dict]] = None,
        train_split_ratio: float = 0.8,
        **kwargs,
    ) -> None:
        self.train_split_ratio = train_split_ratio
        super().__init__(
            data_root=data_root,
            split=split,
            cls_names=cls_names,
            multi_class=multi_class,
            pipeline=pipeline,
            **kwargs,
        )

    def load_data_list(self) -> List[Dict]:
        """Load data annotations for Kolektor.

        Returns:
            List of dicts with keys: img_path, gt_label, gt_mask_path,
            cls_name, defect_type.
        """
        data_list: List[Dict] = []

        # Collect all samples first
        all_samples: List[Dict] = []
        good_samples: List[Dict] = []
        bad_samples: List[Dict] = []

        # Walk through item directories (kos01, kos02, etc.)
        for item_name in sorted(os.listdir(self.data_root)):
            item_dir = osp.join(self.data_root, item_name)
            if not osp.isdir(item_dir):
                continue

            # Find all image files and their corresponding masks
            for img_name in sorted(os.listdir(item_dir)):
                if not img_name.lower().endswith('.jpg'):
                    continue

                img_path = osp.join(item_dir, img_name)
                stem = osp.splitext(img_name)[0]
                mask_name = f'{stem}_label.bmp'
                mask_path = osp.join(item_dir, mask_name)

                if not osp.exists(mask_path):
                    continue

                # Determine if anomalous by reading mask
                is_anomalous = _is_mask_anomalous(mask_path)
                gt_label = 1 if is_anomalous else 0

                sample = dict(
                    img_path=img_path,
                    gt_label=gt_label,
                    gt_mask_path=mask_path if is_anomalous else '',
                    cls_name='kolektor',
                    defect_type='bad' if is_anomalous else 'good',
                )

                if is_anomalous:
                    bad_samples.append(sample)
                else:
                    good_samples.append(sample)

        # Split good samples into train/test (seed=42 for reproducibility)
        np.random.seed(42)
        n_good = len(good_samples)
        indices = np.random.permutation(n_good)
        n_train = int(n_good * self.train_split_ratio)

        train_indices = set(indices[:n_train])

        for i, sample in enumerate(good_samples):
            if i in train_indices:
                sample['_split'] = 'train'
            else:
                sample['_split'] = 'test'

        # All bad samples go to test
        for sample in bad_samples:
            sample['_split'] = 'test'

        # Combine and filter by split
        all_samples = good_samples + bad_samples
        for sample in all_samples:
            if sample['_split'] == self.split:
                # Remove temporary key
                del sample['_split']
                data_list.append(sample)

        return data_list

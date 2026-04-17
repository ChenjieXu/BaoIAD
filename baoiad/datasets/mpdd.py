"""MPDD (Metal Parts Defect Detection) dataset."""

import os
import os.path as osp
from typing import Dict, List

from baoiad.datasets.base_ad_dataset import BaseADDataset
from baoiad.registry import DATASETS


@DATASETS.register_module()
class MPDDDataset(BaseADDataset):
    """MPDD dataset for metal parts defect detection.

    Expected directory structure::

        data_root/
        ├── bracket_black/
        │   ├── train/
        │   │   └── good/
        │   ├── test/
        │   │   ├── good/
        │   │   └── <defect_type>/
        │   └── ground_truth/
        │       └── <defect_type>/
        ├── bracket_brown/
        └── ...

    Reference:
        S. Jezek et al., "Deep learning-based defect detection of metal parts:
        evaluating current methods in complex conditions", ICUMT 2021.

    Args:
        data_root: Path to MPDD root directory.
        split: 'train' or 'test'.
        cls_names: Categories to include. None for all.
        multi_class: Whether to load multiple categories.
        pipeline: Data transform pipeline.
    """

    ALL_CATEGORIES = (
        'bracket_black',
        'bracket_brown',
        'bracket_white',
        'connector',
        'metal_plate',
        'tubes',
    )

    def load_data_list(self) -> List[Dict]:
        """Load data annotations for MPDD.

        Returns:
            List of dicts with keys: img_path, gt_label, gt_mask_path,
            cls_name, defect_type.
        """
        data_list: List[Dict] = []
        for cls_name in self.cls_names:
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
                        mask_name = f'{stem}_mask.png'
                        mask_path = osp.join(gt_dir, defect_type, mask_name)
                        if osp.exists(mask_path):
                            gt_mask_path = mask_path

                    data_list.append(dict(
                        img_path=img_path,
                        gt_label=gt_label,
                        gt_mask_path=gt_mask_path,
                        cls_name=cls_name,
                        defect_type=defect_type,
                    ))

        return data_list

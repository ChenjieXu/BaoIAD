"""BTech (BeanTech Anomaly Detection) dataset."""

import os
import os.path as osp
from typing import Dict, List

from baoiad.datasets.base_ad_dataset import BaseADDataset
from baoiad.registry import DATASETS


@DATASETS.register_module()
class BTechDataset(BaseADDataset):
    """BeanTech Anomaly Detection (BTech/BTAD) dataset.

    Expected directory structure::

        data_root/
        ├── 01/
        │   ├── train/
        │   │   └── ok/
        │   ├── test/
        │   │   ├── ok/
        │   │   └── ko/
        │   └── ground_truth/
        │       └── ko/
        ├── 02/
        ├── 03/
        └── ...

    Note: BTech uses 'ok' instead of 'good' for normal samples,
    and 'ko' for defective samples.

    Reference:
        Mishra et al., "VT-ADL: A Vision Transformer Network for Image
        Anomaly Detection and Localization", ISIE 2021.
    """

    ALL_CATEGORIES = ('01', '02', '03')

    # BTech uses 'ok' for normal class instead of 'good'
    NORMAL_DIR = 'ok'

    def load_data_list(self) -> List[Dict]:
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

                is_normal = (defect_type == self.NORMAL_DIR)
                gt_label = 0 if is_normal else 1

                for img_name in sorted(os.listdir(defect_dir)):
                    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        continue

                    img_path = osp.join(defect_dir, img_name)

                    gt_mask_path = ''
                    if self.split == 'test' and not is_normal:
                        stem = osp.splitext(img_name)[0]
                        # BTech mask naming may vary; try common patterns
                        for mask_pattern in [f'{stem}.png', f'{stem}_mask.png']:
                            mask_path = osp.join(gt_dir, defect_type, mask_pattern)
                            if osp.exists(mask_path):
                                gt_mask_path = mask_path
                                break

                    data_list.append(dict(
                        img_path=img_path,
                        gt_label=gt_label,
                        gt_mask_path=gt_mask_path,
                        cls_name=cls_name,
                        defect_type=defect_type,
                    ))

        return data_list

"""MVTec LOCO AD dataset."""

import os
import os.path as osp
from typing import Dict, List

from baoiad.datasets.base_ad_dataset import BaseADDataset
from baoiad.registry import DATASETS


@DATASETS.register_module()
class MVTecLOCODataset(BaseADDataset):
    """MVTec Logical Constraints Anomaly Detection (LOCO AD) dataset.

    Expected directory structure::

        data_root/
        ├── breakfast_box/
        │   ├── train/
        │   │   └── good/
        │   ├── validation/
        │   │   └── good/
        │   ├── test/
        │   │   ├── good/
        │   │   ├── logical_anomalies/
        │   │   └── structural_anomalies/
        │   └── ground_truth/
        │       ├── logical_anomalies/
        │       │   ├── 000_mask.png
        │       │   └── ...
        │       └── structural_anomalies/
        └── ...

    Note: This dataset contains both structural and logical anomalies.
    The 'validation' split uses validation/good/ images.

    Reference:
        Bergmann et al., "Beyond Dents and Scratches: Logical Constraints
        in Unsupervised Anomaly Detection and Localization", IJCV 2022.
    """

    ALL_CATEGORIES = (
        'breakfast_box', 'juice_bottle', 'pushpins',
        'screw_bag', 'splicing_connectors',
    )

    VALID_SPLITS = ('train', 'test', 'validation')

    def load_data_list(self) -> List[Dict]:
        data_list: List[Dict] = []
        assert self.split in self.VALID_SPLITS, (
            f"Invalid split '{self.split}', expected one of {self.VALID_SPLITS}"
        )
        for cls_name in self.cls_names:
            split = self.split
            cls_dir = osp.join(self.data_root, cls_name, split)
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
                    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        continue

                    img_path = osp.join(defect_dir, img_name)

                    gt_mask_path = ''
                    if self.split == 'test' and not is_normal:
                        stem = osp.splitext(img_name)[0]
                        # Try multiple mask naming conventions
                        for mask_pattern in [f'{stem}_mask.png', f'{stem}.png']:
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

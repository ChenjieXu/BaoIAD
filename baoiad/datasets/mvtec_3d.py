"""MVTec 3D-AD dataset (RGB only)."""

import os
import os.path as osp
from typing import Dict, List

from baoiad.datasets.base_ad_dataset import BaseADDataset
from baoiad.registry import DATASETS


@DATASETS.register_module()
class MVTec3DDataset(BaseADDataset):
    """MVTec 3D Anomaly Detection dataset (RGB only).

    Expected directory structure::

        data_root/
        ├── bagel/
        │   ├── train/
        │   │   └── good/
        │   │       └── rgb/
        │   │           ├── 000.png
        │   │           └── ...
        │   ├── test/
        │   │   ├── good/
        │   │   │   └── rgb/
        │   │   ├── crack/
        │   │   │   ├── rgb/
        │   │   │   └── gt/
        │   │   │       ├── 000.png
        │   │   │       └── ...
        │   │   └── ...
        ├── cable_gland/
        └── ...

    Note: Only RGB images are used. 3D point clouds and organized
    point clouds are ignored.

    Reference:
        Bergmann et al., "The MVTec 3D-AD Dataset for Unsupervised 3D
        Anomaly Detection and Localization", VISAPP 2022.
    """

    ALL_CATEGORIES = (
        'bagel', 'cable_gland', 'carrot', 'cookie', 'dowel',
        'foam', 'peach', 'potato', 'rope', 'tire',
    )

    def load_data_list(self) -> List[Dict]:
        data_list: List[Dict] = []
        for cls_name in self.cls_names:
            cls_dir = osp.join(self.data_root, cls_name, self.split)
            if not osp.isdir(cls_dir):
                continue

            for defect_type in sorted(os.listdir(cls_dir)):
                defect_dir = osp.join(cls_dir, defect_type)
                if not osp.isdir(defect_dir):
                    continue

                is_normal = (defect_type == 'good')
                gt_label = 0 if is_normal else 1

                # MVTec 3D-AD stores RGB images in a 'rgb' subdirectory
                rgb_dir = osp.join(defect_dir, 'rgb')
                if not osp.isdir(rgb_dir):
                    # Fallback: images directly in defect_dir
                    rgb_dir = defect_dir

                # Ground truth masks are in test/<defect>/gt/
                gt_dir = osp.join(defect_dir, 'gt')

                for img_name in sorted(os.listdir(rgb_dir)):
                    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        continue

                    img_path = osp.join(rgb_dir, img_name)

                    gt_mask_path = ''
                    if self.split == 'test' and not is_normal and osp.isdir(gt_dir):
                        stem = osp.splitext(img_name)[0]
                        for mask_pattern in [f'{stem}.png', f'{stem}_mask.png']:
                            mask_path = osp.join(gt_dir, mask_pattern)
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

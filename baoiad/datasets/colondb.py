"""ColonDB dataset with MVTec-like folder layout."""

import os
import os.path as osp
from typing import Dict, List

from baoiad.datasets.base_ad_dataset import BaseADDataset
from baoiad.registry import DATASETS


@DATASETS.register_module()
class ColonDBDataset(BaseADDataset):
    """ColonDB dataset stored in a MVTec-like folder layout.

    Accepts both `<stem>_mask.png` and `<stem>.png` ground-truth mask names.
    """

    ALL_CATEGORIES = ('ColonDB',)

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

                is_normal = defect_type == 'good'
                gt_label = 0 if is_normal else 1

                for img_name in sorted(os.listdir(defect_dir)):
                    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                        continue

                    img_path = osp.join(defect_dir, img_name)
                    gt_mask_path = ''
                    if self.split == 'test' and not is_normal:
                        stem = osp.splitext(img_name)[0]
                        mask_dir = osp.join(gt_dir, defect_type)
                        for mask_name in (f'{stem}_mask.png', f'{stem}.png', f'{stem}.bmp', f'{stem}.jpg'):
                            candidate = osp.join(mask_dir, mask_name)
                            if osp.exists(candidate):
                                gt_mask_path = candidate
                                break

                    data_list.append(
                        dict(
                            img_path=img_path,
                            gt_label=gt_label,
                            gt_mask_path=gt_mask_path,
                            cls_name=cls_name,
                            defect_type=defect_type,
                        )
                    )
        return data_list

"""VisA (Visual Anomaly) dataset."""

import csv
import os
import os.path as osp
from typing import Dict, List, Optional

from baoiad.datasets.base_ad_dataset import BaseADDataset
from baoiad.registry import DATASETS


@DATASETS.register_module(force=True)
class VisADataset(BaseADDataset):
    """Visual Anomaly (VisA) dataset.

    Expected directory structure::

        data_root/
        ├── candle/
        │   ├── Data/
        │   │   ├── Images/
        │   │   │   ├── Normal/
        │   │   │   │   ├── 0000.JPG
        │   │   │   │   └── ...
        │   │   │   └── Anomaly/
        │   │   │       ├── 000.JPG
        │   │   │       └── ...
        │   │   └── Masks/
        │   │       └── Anomaly/
        │   │           ├── 000.png
        │   │           └── ...
        │   └── image_anno.csv
        ├── split_csv/
        │   └── 1cls.csv
        ├── capsules/
        └── ...

    The dataset uses the official 1cls split CSV for train/test splits.

    Reference:
        Zou et al., "SPot-the-Difference Self-supervised Pre-training for
        Anomaly Detection and Segmentation", ECCV 2022.
    """

    ALL_CATEGORIES = (
        'candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
        'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4',
        'pipe_fryum',
    )

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        cls_names: Optional[List[str]] = None,
        multi_class: bool = True,
        pipeline: Optional[List[dict]] = None,
        split_file: str = 'split_csv/1cls.csv',
        **kwargs,
    ) -> None:
        self.split_file = split_file
        super().__init__(
            data_root=data_root,
            split=split,
            cls_names=cls_names,
            multi_class=multi_class,
            pipeline=pipeline,
            **kwargs,
        )

    def load_data_list(self) -> List[Dict]:
        csv_path = osp.join(self.data_root, self.split_file)
        if osp.exists(csv_path):
            return self._load_from_official_csv(csv_path)
        return self._load_from_mvt_like_layout()

    def _load_from_official_csv(self, csv_path: str) -> List[Dict]:
        data_list: List[Dict] = []
        cls_set = set(self.cls_names)

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                cls_name = row['object']
                if cls_name not in cls_set:
                    continue

                row_split = row['split']
                if row_split != self.split:
                    continue

                label = row['label']
                is_normal = (label == 'normal')
                gt_label = 0 if is_normal else 1

                img_path = osp.join(self.data_root, row['image'])

                gt_mask_path = ''
                mask_rel = row.get('mask', '').strip()
                if mask_rel:
                    gt_mask_path = osp.join(self.data_root, mask_rel)

                defect_type = 'good' if is_normal else 'anomaly'

                data_list.append(dict(
                    img_path=img_path,
                    gt_label=gt_label,
                    gt_mask_path=gt_mask_path,
                    cls_name=cls_name,
                    defect_type=defect_type,
                ))

        return data_list

    def _load_from_mvt_like_layout(self) -> List[Dict]:
        data_list: List[Dict] = []
        valid_suffixes = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.JPG', '.PNG')

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
                    if not img_name.lower().endswith(tuple(s.lower() for s in valid_suffixes)):
                        continue

                    img_path = osp.join(defect_dir, img_name)
                    gt_mask_path = ''
                    if not is_normal:
                        stem = osp.splitext(img_name)[0]
                        mask_dir = osp.join(gt_dir, defect_type)
                        candidates = [
                            osp.join(mask_dir, f'{stem}_mask.png'),
                            osp.join(mask_dir, f'{stem}.png'),
                            osp.join(mask_dir, f'{stem}.bmp'),
                            osp.join(mask_dir, f'{stem}.jpg'),
                        ]
                        for candidate in candidates:
                            if osp.exists(candidate):
                                gt_mask_path = candidate
                                break

                    data_list.append(dict(
                        img_path=img_path,
                        gt_label=gt_label,
                        gt_mask_path=gt_mask_path,
                        cls_name=cls_name,
                        defect_type=defect_type,
                    ))

        return data_list

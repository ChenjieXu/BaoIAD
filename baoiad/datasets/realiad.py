"""Real-IAD dataset."""

import json
import os.path as osp
from typing import Dict, List, Optional

from baoiad.datasets.base_ad_dataset import BaseADDataset
from baoiad.registry import DATASETS


@DATASETS.register_module(force=True)
class RealIADDataset(BaseADDataset):
    """Real-IAD dataset for real-world industrial anomaly detection.

    Expected directory structure::

        data_root/
        ├── realiad_256/
        │   └── audiojack/
        │       ├── OK/
        │       │   └── S0001/
        │       │       └── audiojack_0001_OK_C1_xxx.jpg
        │       └── NG/
        │           └── <defect_type>/
        │               └── S0002/
        │                   ├── audiojack_0002_NG_C1_xxx.jpg
        │                   └── audiojack_0002_NG_C1_xxx_mask.png
        ├── realiad_512/
        ├── realiad_1024/
        └── realiad_jsons/
            └── realiad_jsons/
                └── audiojack.json

    Reference:
        Real-IAD: A Real-World Multi-View Industrial Anomaly Detection Dataset

    Args:
        data_root: Path to Real-IAD root directory.
        split: 'train' or 'test'.
        cls_names: Categories to include. None for all.
        multi_class: Whether to load multiple categories.
        pipeline: Data transform pipeline.
        resolution: Image resolution ('256', '512', '1024', or 'raw').
        json_path: Path to JSON metadata directory, relative to data_root.
            Default uses multi-view metadata.
    """

    ALL_CATEGORIES = (
        'audiojack',
        'bottle_cap',
        'button_battery',
        'end_cap',
        'eraser',
        'fire_hood',
        'mint',
        'mounts',
        'pcb',
        'phone_battery',
        'plastic_nut',
        'plastic_plug',
        'porcelain_doll',
        'regulator',
        'rolled_strip_base',
        'sim_card_set',
        'switch',
        'tape',
        'terminalblock',
        'toothbrush',
        'toy_brick',
        'toy',
        'transistor1',
        'u_block',
        'usb_adaptor',
        'usb',
        'vcpill',
        'wooden_beads',
        'woodstick',
        'zipper',
    )

    RESOLUTIONS = ('256', '512', '1024', 'raw')

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        cls_names: Optional[List[str]] = None,
        multi_class: bool = True,
        pipeline: Optional[List[dict]] = None,
        resolution: str = '256',
        json_path: str = 'realiad_jsons/realiad_jsons',
        **kwargs,
    ) -> None:
        # Validate resolution
        if str(resolution) not in self.RESOLUTIONS:
            raise ValueError(
                f"Invalid resolution '{resolution}'. "
                f"Must be one of {self.RESOLUTIONS}"
            )
        self.resolution = str(resolution)
        self.json_path = json_path
        super().__init__(
            data_root=data_root,
            split=split,
            cls_names=cls_names,
            multi_class=multi_class,
            pipeline=pipeline,
            **kwargs,
        )

    def load_data_list(self) -> List[Dict]:
        """Load data annotations for Real-IAD from JSON metadata.

        Returns:
            List of dicts with keys: img_path, gt_label, gt_mask_path,
            cls_name, defect_type.
        """
        data_list: List[Dict] = []

        for cls_name in self.cls_names:
            # Load JSON metadata for this category
            json_file = osp.join(self.data_root, self.json_path, f'{cls_name}.json')
            if not osp.exists(json_file):
                continue

            with open(json_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # Get samples for the requested split
            split_key = self.split
            if split_key not in metadata:
                continue

            samples = metadata[split_key]

            # Build path to category directory
            category_dir = osp.join(
                self.data_root,
                f'realiad_{self.resolution}',
                cls_name
            )

            for sample in samples:
                img_rel_path = sample.get('image_path', '')
                mask_rel_path = sample.get('mask_path', '')
                anomaly_class = sample.get('anomaly_class', 'OK')

                if not img_rel_path:
                    continue

                img_path = osp.join(category_dir, img_rel_path)
                gt_mask_path = ''
                if mask_rel_path:
                    gt_mask_path = osp.join(category_dir, mask_rel_path)

                gt_label = 0 if anomaly_class == 'OK' else 1
                defect_type = 'good' if gt_label == 0 else anomaly_class

                data_list.append(dict(
                    img_path=img_path,
                    gt_label=gt_label,
                    gt_mask_path=gt_mask_path,
                    cls_name=cls_name,
                    defect_type=defect_type,
                ))

        return data_list

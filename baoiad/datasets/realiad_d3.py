"""RealIAD D3 dataset for BaoIAD."""

import json
import os.path as osp
from typing import Dict, List, Optional

from baoiad.datasets.base_ad_dataset import BaseADDataset
from baoiad.registry import DATASETS


@DATASETS.register_module(force=True)
class RealIADD3Dataset(BaseADDataset):
    """RealIAD D3 dataset for industrial anomaly detection.

    Reads from meta.json generated for D3. Each class has ~50 training
    (OK only) and ~400-500 test (OK + NG) samples.

    Args:
        data_root: Path to D3 root (contains meta.json and class dirs).
        split: 'train' or 'test'.
        cls_names: Categories to include. None for all 20 classes.
        multi_class: Whether to load multiple categories.
        pipeline: Data transform pipeline configs.
    """

    ALL_CATEGORIES = (
        'audio_jack_socket', 'common_mode_filter', 'connector_housing_female',
        'crimp_st_cable_mount_box', 'dc_power_connector', 'ethernet_connector',
        'ferrite_bead', 'fork_crimp_terminal', 'fuse_holder', 'headphone_jack_socket',
        'humidity_sensor', 'knob_cap', 'lattice_block_plug', 'lego_pin_connector_plate',
        'lego_propeller', 'limit_switch', 'miniature_lifting_motor', 'power_jack',
        'purple_clay_pot', 'telephone_spring_switch',
    )

    def load_data_list(self) -> List[Dict]:
        data_list: List[Dict] = []
        meta_path = osp.join(self.data_root, 'meta.json')
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        split_data = meta.get(self.split, {})
        for cls_name in self.cls_names:
            if cls_name not in split_data:
                continue
            for entry in split_data[cls_name]:
                img_path = osp.join(self.data_root, entry['img_path'])
                gt_label = int(entry.get('anomaly', 0))

                # Mask handling
                mask_path = entry.get('mask_path', '')
                if mask_path:
                    full_mask = osp.join(self.data_root, mask_path)
                    gt_mask_path = full_mask if osp.exists(full_mask) else ''
                else:
                    gt_mask_path = ''

                defect_type = entry.get('specie_name', 'good')
                if gt_label == 0:
                    defect_type = 'good'

                data_list.append(dict(
                    img_path=img_path,
                    gt_label=gt_label,
                    gt_mask_path=gt_mask_path,
                    cls_name=cls_name,
                    defect_type=defect_type,
                ))

        return data_list

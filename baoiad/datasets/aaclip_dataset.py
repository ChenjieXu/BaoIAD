"""AA-CLIP datasets aligned to the official jsonl metadata pipeline."""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional

import torch
from mmengine.dataset import BaseDataset
from PIL import Image
from torchvision import transforms

from baoiad.registry import DATASETS
from baoiad.structures import ADDataSample


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

CLASS_NAMES = {
    'MVTec': [
        'bottle', 'cable', 'capsule', 'carpet', 'grid',
        'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
        'tile', 'transistor', 'toothbrush', 'wood', 'zipper',
    ],
    'VisA': [
        'candle', 'pcb3', 'capsules', 'pipe_fryum', 'pcb4', 'macaroni2',
        'pcb2', 'chewinggum', 'macaroni1', 'cashew', 'fryum', 'pcb1',
    ],
}


@DATASETS.register_module()
class AACLIPJsonDataset(BaseDataset):
    """AA-CLIP dataset that reads official-style jsonl metadata."""

    METAINFO: dict = dict(task='anomaly_detection')

    def __init__(
        self,
        data_root: str,
        metadata_path: str,
        dataset_name: str,
        img_size: int = 518,
        cls_names: Optional[List[str]] = None,
        multi_class: bool = True,
        text_mode: bool = False,
        augment: bool = False,
        pipeline: Optional[List[dict]] = None,
        **kwargs,
    ) -> None:
        self.metadata_path = metadata_path
        self.dataset_name = dataset_name
        self.img_size = int(img_size)
        self.multi_class = bool(multi_class)
        self.text_mode = bool(text_mode)
        self.augment = bool(augment)

        if cls_names is not None:
            self.cls_names = list(cls_names)
        elif multi_class:
            self.cls_names = list(CLASS_NAMES.get(dataset_name, []))
        else:
            raise ValueError('`cls_names` must be provided when `multi_class=False`.')

        transform_x = []
        if self.augment and not self.text_mode:
            transform_x.extend([
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.5)],
                    p=0.7,
                ),
                transforms.RandomApply(
                    [transforms.ColorJitter(contrast=0.5)],
                    p=0.7,
                ),
                transforms.RandomApply(
                    [transforms.ColorJitter(saturation=0.5)],
                    p=0.7,
                ),
            ])

        self.transform_x = transforms.Compose(
            transform_x + [
                transforms.Resize((self.img_size, self.img_size), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
            ]
        )
        self.transform_mask = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), Image.NEAREST),
            transforms.ToTensor(),
        ])
        self.random_transform = transforms.Compose([
            transforms.RandomApply(
                [transforms.RandomRotation(degrees=math.degrees(math.pi / 6))],
                p=0.5,
            ),
            transforms.RandomApply(
                [transforms.RandomAffine(degrees=0, translate=(0.15, 0.15))],
                p=0.5,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ])

        super().__init__(data_root=data_root, pipeline=pipeline or [], **kwargs)

    def load_data_list(self) -> List[Dict]:
        allowed_classes = set(self.cls_names)
        data_list: List[Dict] = []
        with open(self.metadata_path, 'r', encoding='utf-8') as handle:
            for line in handle:
                meta = json.loads(line)
                class_name = str(meta['class_name'])
                if class_name not in allowed_classes:
                    continue

                # Remap official VisA paths to local BaoIAD format
                image_path = meta['image_path']
                mask_path = meta.get('mask_path', '')

                # Handle official VisA dataset structure -> local BaoIAD structure
                if '/Data/Images/Normal/' in image_path:
                    image_path = image_path.replace('/Data/Images/Normal/', '/test/good/')
                elif '/Data/Images/Anomaly/' in image_path:
                    image_path = image_path.replace('/Data/Images/Anomaly/', '/test/bad/')

                if mask_path and '/Data/Masks/Anomaly/' in mask_path:
                    mask_path = mask_path.replace('/Data/Masks/Anomaly/', '/ground_truth/bad/')

                data_list.append(dict(
                    img_path=os.path.join(self.data_root, image_path),
                    gt_mask_path=os.path.join(self.data_root, mask_path)
                    if mask_path
                    else '',
                    gt_label=int(meta['label']),
                    cls_name=class_name,
                    defect_type='good' if int(meta['label']) == 0 else 'anomaly',
                ))
        return data_list

    def __getitem__(self, idx: int) -> Dict:
        if not self._fully_initialized:
            self.full_init()

        data_info = self.get_data_info(idx)
        img = Image.open(data_info['img_path']).convert('RGB')
        img = self.transform_x(img)

        if data_info['gt_label'] and data_info['gt_mask_path']:
            mask = Image.open(data_info['gt_mask_path']).convert('L')
            mask = self.transform_mask(mask)
            mask = (mask != 0).float()
        else:
            mask = torch.zeros(1, self.img_size, self.img_size)

        if self.augment:
            transform_tensor = torch.cat([img, mask], dim=0)
            transform_tensor = self.random_transform(transform_tensor)
            img = transform_tensor[:3]
            mask = transform_tensor[3:4]

        data_sample = ADDataSample()
        data_sample.set_metainfo({
            'cls_name': data_info['cls_name'],
            'img_path': data_info['img_path'],
            'defect_type': data_info['defect_type'],
        })
        data_sample.gt_label = int(data_info['gt_label'])
        data_sample.gt_mask = mask.squeeze(0)
        return dict(inputs=img, data_samples=data_sample)

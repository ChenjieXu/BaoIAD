"""AdaCLIP strict auxiliary datasets.

These loaders mirror the official AdaCLIP dataset semantics more closely than
the generic benchmark datasets:

- prefer ``meta.json`` when it exists
- always consume the labeled ``test`` partition used by the official repo for
  both training and evaluation
- fall back to the local MVTec-like layout only when no ``meta.json`` is
  available locally
"""

from __future__ import annotations

import json
import os
import os.path as osp
from typing import Dict, Iterable, List, Sequence

from baoiad.datasets.clinicdb import ClinicDBDataset
from baoiad.datasets.colondb import ColonDBDataset
from baoiad.datasets.visa import VisADataset
from baoiad.registry import DATASETS


def _load_adaclip_meta(
    data_root: str,
    cls_names: Sequence[str],
    *,
    split: str = 'test',
) -> List[Dict]:
    meta_path = osp.join(data_root, 'meta.json')
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta_info = json.load(f)

    split_info = meta_info.get(split, {})
    data_list: List[Dict] = []
    for cls_name in cls_names:
        for item in split_info.get(cls_name, []):
            gt_label = int(item.get('anomaly', 0))
            mask_rel = item.get('mask_path', '') or ''
            gt_mask_path = osp.join(data_root, mask_rel) if mask_rel else ''
            resolved_cls_name = item.get('cls_name', cls_name)
            defect_type = 'good' if gt_label == 0 else item.get('specie_name', 'anomaly')
            data_list.append(
                dict(
                    img_path=osp.join(data_root, item['img_path']),
                    gt_label=gt_label,
                    gt_mask_path=gt_mask_path,
                    cls_name=resolved_cls_name,
                    defect_type=defect_type,
                )
            )
    return data_list


def _iter_valid_images(defect_dir: str) -> Iterable[str]:
    suffixes = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.JPG', '.PNG')
    for img_name in sorted(os.listdir(defect_dir)):
        if img_name.lower().endswith(tuple(s.lower() for s in suffixes)):
            yield img_name


def _load_single_class_test_layout(
    data_root: str,
    cls_names: Sequence[str],
    *,
    good_dir_names: Sequence[str] = ('good',),
) -> List[Dict]:
    data_list: List[Dict] = []
    for cls_name in cls_names:
        cls_dir = osp.join(data_root, cls_name, 'test')
        if not osp.isdir(cls_dir):
            continue

        gt_dir = osp.join(data_root, cls_name, 'ground_truth')
        for defect_type in sorted(os.listdir(cls_dir)):
            defect_dir = osp.join(cls_dir, defect_type)
            if not osp.isdir(defect_dir):
                continue

            is_normal = defect_type in set(good_dir_names)
            gt_label = 0 if is_normal else 1

            for img_name in _iter_valid_images(defect_dir):
                img_path = osp.join(defect_dir, img_name)
                gt_mask_path = ''
                if not is_normal:
                    stem = osp.splitext(img_name)[0]
                    mask_dir = osp.join(gt_dir, defect_type)
                    for mask_name in (
                        f'{stem}_mask.png',
                        f'{stem}.png',
                        f'{stem}.bmp',
                        f'{stem}.jpg',
                    ):
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
                        defect_type='good' if is_normal else defect_type,
                    )
                )
    return data_list


class _AdaCLIPStrictDatasetMixin:
    """Force official AdaCLIP's labeled-test-partition semantics."""

    OFFICIAL_SPLIT = 'test'

    def load_data_list(self) -> List[Dict]:
        meta_path = osp.join(self.data_root, 'meta.json')
        if osp.exists(meta_path):
            return _load_adaclip_meta(
                self.data_root,
                self.cls_names,
                split=self.OFFICIAL_SPLIT,
            )
        return self._load_official_fallback()

    def _load_official_fallback(self) -> List[Dict]:
        raise NotImplementedError


@DATASETS.register_module()
class AdaCLIPVisADataset(_AdaCLIPStrictDatasetMixin, VisADataset):
    """VisA loader for AdaCLIP strict alignment."""

    def _load_official_fallback(self) -> List[Dict]:
        original_split = self.split
        self.split = self.OFFICIAL_SPLIT
        try:
            return self._load_from_mvt_like_layout()
        finally:
            self.split = original_split


@DATASETS.register_module()
class AdaCLIPClinicDBDataset(_AdaCLIPStrictDatasetMixin, ClinicDBDataset):
    """ClinicDB loader for AdaCLIP strict alignment."""

    def _load_official_fallback(self) -> List[Dict]:
        return _load_single_class_test_layout(self.data_root, self.cls_names)


@DATASETS.register_module()
class AdaCLIPColonDBDataset(_AdaCLIPStrictDatasetMixin, ColonDBDataset):
    """ColonDB loader for AdaCLIP strict alignment."""

    def _load_official_fallback(self) -> List[Dict]:
        return _load_single_class_test_layout(self.data_root, self.cls_names)

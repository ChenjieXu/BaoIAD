"""Tests for NSA training dataset."""

from pathlib import Path

import cv2
import numpy as np

import baoiad  # noqa: F401
from baoiad.datasets import nsa_dataset as nsa_dataset_module
from baoiad.datasets.nsa_dataset import NSATrainDataset


def _write_rgb(path: Path, value: int) -> None:
    image = np.full((32, 32, 3), value, dtype=np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def _build_train_tree(root: Path, cls_name: str, num_images: int = 3) -> None:
    train_good = root / cls_name / 'train' / 'good'
    train_good.mkdir(parents=True, exist_ok=True)
    for idx in range(num_images):
        _write_rgb(train_good / f'{idx:03d}.png', value=40 + idx * 20)


def test_prev_index_updates_between_samples(tmp_path, monkeypatch):
    _build_train_tree(tmp_path, 'bottle', num_images=3)

    def fake_patch_ex(**kwargs):
        dest = kwargs['ima_dest']
        label = np.zeros(dest.shape[:2] + (1,), dtype=np.float32)
        return dest.copy(), label

    monkeypatch.setattr(nsa_dataset_module, '_official_patch_ex', fake_patch_ex)

    dataset = NSATrainDataset(
        data_root=str(tmp_path),
        cls_names=['bottle'],
        anomaly_ratio=1.0,
        pipeline=[],
        lazy_init=False,
    )

    dataset._prev_index_by_class['bottle'] = 0
    sample1 = dataset[1]
    sample2 = dataset[2]

    assert sample1['source_index'] == 0
    assert sample2['source_index'] == 1
    assert sample1['source_img_path'].endswith('000.png')
    assert sample2['source_img_path'].endswith('001.png')


def test_object_vs_texture_paths(tmp_path, monkeypatch):
    _build_train_tree(tmp_path, 'bottle', num_images=2)
    _build_train_tree(tmp_path, 'tile', num_images=2)

    record = []

    def fake_patch_ex(**kwargs):
        record.append((kwargs['mode'], kwargs['ima_dest'].shape[:2]))
        dest = kwargs['ima_dest']
        label = np.zeros(dest.shape[:2] + (1,), dtype=np.float32)
        return dest.copy(), label

    monkeypatch.setattr(nsa_dataset_module, '_official_patch_ex', fake_patch_ex)

    bottle_ds = NSATrainDataset(
        data_root=str(tmp_path),
        cls_names=['bottle'],
        anomaly_ratio=1.0,
        pipeline=[],
        lazy_init=False,
    )
    tile_ds = NSATrainDataset(
        data_root=str(tmp_path),
        cls_names=['tile'],
        anomaly_ratio=1.0,
        pipeline=[],
        lazy_init=False,
    )

    _ = bottle_ds[0]
    _ = tile_ds[0]

    assert record[0][0] == nsa_dataset_module.CV2_NORMAL_CLONE
    assert record[0][1] == (224, 224)
    assert record[1][0] == nsa_dataset_module.CV2_MIXED_CLONE
    assert record[1][1] == (256, 256)


def test_anomaly_ratio_zero_returns_clean_sample(tmp_path):
    _build_train_tree(tmp_path, 'bottle', num_images=2)

    dataset = NSATrainDataset(
        data_root=str(tmp_path),
        cls_names=['bottle'],
        anomaly_ratio=0.0,
        pipeline=[],
        lazy_init=False,
    )

    sample = dataset[0]
    assert sample['has_anomaly'] == 0.0
    assert sample['gt_label'] == 0
    assert sample['gt_mask'].shape == (224, 224)
    assert np.allclose(sample['gt_mask'], 0.0)
    assert sample['img'].shape == (224, 224, 3)

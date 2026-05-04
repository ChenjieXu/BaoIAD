"""Tests for official-compatible RegAD datasets."""

from pathlib import Path

import cv2
import torch

import baoiad  # noqa: F401

from baoiad.datasets.regad_dataset import RegADTestDataset, RegADTrainDataset
from baoiad.datasets.transforms import PackADInputs


ALL_CATEGORIES = (
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper',
)


def _write_rgb(path: Path, value: int) -> None:
    img = (torch.full((8, 8, 3), value, dtype=torch.uint8)).numpy()
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def _write_mask(path: Path) -> None:
    mask = torch.zeros((8, 8), dtype=torch.uint8)
    mask[2:6, 2:6] = 255
    cv2.imwrite(str(path), mask.numpy())


def _build_train_root(root: Path) -> None:
    for idx, cls_name in enumerate(ALL_CATEGORIES):
        train_dir = root / cls_name / 'train' / 'good'
        train_dir.mkdir(parents=True, exist_ok=True)
        _write_rgb(train_dir / '000.png', 20 + idx)
        _write_rgb(train_dir / '001.png', 40 + idx)


def test_regad_train_dataset_returns_query_and_support(tmp_path):
    data_root = tmp_path / 'mvtec'
    _build_train_root(data_root)

    dataset = RegADTrainDataset(
        data_root=str(data_root),
        target_cls='bottle',
        img_size=8,
        shot=2,
        pipeline=[PackADInputs()],
    )

    packed = dataset[0]
    sample = packed['data_samples']
    assert isinstance(packed['inputs'], torch.Tensor)
    assert packed['inputs'].shape == (3, 8, 8)
    assert hasattr(sample, 'support_imgs')
    assert sample.support_imgs.shape == (2, 3, 8, 8)
    assert sample.source_cls != 'bottle'
    assert sample.target_cls == 'bottle'
    assert 0.0 <= float(packed['inputs'].min().item()) <= 1.0
    assert 0.0 <= float(packed['inputs'].max().item()) <= 1.0


def test_regad_train_shuffle_dataset_resamples_supports(tmp_path):
    data_root = tmp_path / 'mvtec'
    _build_train_root(data_root)

    dataset = RegADTrainDataset(
        data_root=str(data_root),
        target_cls='bottle',
        img_size=8,
        shot=1,
        pipeline=[PackADInputs()],
    )

    before = [tuple(item['support_img_paths']) for item in dataset.data_list[:5]]
    dataset.shuffle_dataset()
    after = [tuple(item['support_img_paths']) for item in dataset.data_list[:5]]
    assert len(before) == len(after)


def test_regad_test_dataset_loads_masks(tmp_path):
    data_root = tmp_path / 'mvtec'
    _build_train_root(data_root)

    test_good = data_root / 'bottle' / 'test' / 'good'
    test_bad = data_root / 'bottle' / 'test' / 'broken_large'
    gt_dir = data_root / 'bottle' / 'ground_truth' / 'broken_large'
    test_good.mkdir(parents=True, exist_ok=True)
    test_bad.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    _write_rgb(test_good / '000.png', 10)
    _write_rgb(test_bad / '001.png', 200)
    _write_mask(gt_dir / '001_mask.png')

    dataset = RegADTestDataset(
        data_root=str(data_root),
        target_cls='bottle',
        split='test',
        img_size=8,
        pipeline=[PackADInputs()],
    )

    anomalous = dataset[0]
    sample = anomalous['data_samples']
    assert sample.cls_name == 'bottle'
    assert sample.gt_label == 1
    assert sample.gt_mask.shape == (8, 8)
    assert float(sample.gt_mask.sum().item()) > 0.0

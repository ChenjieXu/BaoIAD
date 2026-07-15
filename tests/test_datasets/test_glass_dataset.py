"""Tests for the strict GLASS dataset."""

import importlib
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.optional
pd = pytest.importorskip("pandas", reason='requires the "glass" optional extra')
pytest.importorskip("openpyxl", reason='requires the "glass" optional extra')

importlib.import_module("baoiad")
GLASSDataset = importlib.import_module("baoiad.datasets.glass_dataset").GLASSDataset
transforms = importlib.import_module("baoiad.datasets.transforms")
PackADInputs = transforms.PackADInputs
PackGLASSInputs = transforms.PackGLASSInputs


def _write_rgb_image(path: Path, rgb) -> None:
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    img[..., 0] = rgb[2]
    img[..., 1] = rgb[1]
    img[..., 2] = rgb[0]
    cv2.imwrite(str(path), img)


def _write_mask(path: Path, value: int = 255) -> None:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = value
    cv2.imwrite(str(path), mask)


def _write_distribution_xlsx(path: Path) -> None:
    pd.DataFrame(
        [
            {"Class": "mvtec_bottle", "Distribution": 1, "Foreground": 1},
        ]
    ).to_excel(path, index=False)


def _write_distribution_xlsx_with_rows(path: Path, rows) -> None:
    pd.DataFrame(rows).to_excel(path, index=False)


def test_glass_dataset_train_pack_contains_aug_and_mask(tmp_path, monkeypatch):
    data_root = tmp_path / "mvtec"
    train_dir = data_root / "bottle" / "train" / "good"
    train_dir.mkdir(parents=True)
    _write_rgb_image(train_dir / "000.png", (10, 20, 30))

    fg_dir = tmp_path / "fg_mask" / "bottle"
    fg_dir.mkdir(parents=True)
    _write_mask(fg_dir / "000.png")

    dtd_root = tmp_path / "dtd" / "images" / "banded"
    dtd_root.mkdir(parents=True)
    _write_rgb_image(dtd_root / "tex.png", (50, 100, 150))

    meta_path = tmp_path / "mvtec_distribution.xlsx"
    _write_distribution_xlsx(meta_path)

    import baoiad.datasets.glass_dataset as glass_dataset_module

    monkeypatch.setattr(
        glass_dataset_module,
        "generate_glass_perlin_masks",
        lambda **kwargs: (
            np.pad(np.ones((2, 2), dtype=np.float32), ((1, 1), (1, 1))),
            np.pad(np.ones((4, 4), dtype=np.float32), ((2, 2), (2, 2))),
        ),
    )

    dataset = GLASSDataset(
        data_root=str(data_root),
        split="train",
        cls_names=["bottle"],
        multi_class=False,
        img_size=8,
        resize=8,
        downsampling=2,
        dtd_path=str(dtd_root.parent.parent),
        fg_mask_root=str(tmp_path / "fg_mask"),
        distribution_meta_path=str(meta_path),
        distribution=0,
        pipeline=[PackGLASSInputs()],
    )

    packed = dataset[0]
    assert isinstance(packed["inputs"], torch.Tensor)
    assert packed["inputs"].shape == (3, 8, 8)
    sample = packed["data_samples"]
    assert hasattr(sample, "aug")
    assert hasattr(sample, "mask_s")
    assert sample.aug.shape == (3, 8, 8)
    assert sample.mask_s.shape == (4, 4)
    assert sample.gt_label == 0


def test_glass_dataset_test_pack_contains_gt_mask(tmp_path):
    data_root = tmp_path / "mvtec"
    test_good_dir = data_root / "bottle" / "test" / "good"
    test_bad_dir = data_root / "bottle" / "test" / "broken_small"
    gt_dir = data_root / "bottle" / "ground_truth" / "broken_small"
    test_good_dir.mkdir(parents=True)
    test_bad_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)

    _write_rgb_image(test_good_dir / "000.png", (10, 20, 30))
    _write_rgb_image(test_bad_dir / "001.png", (30, 40, 50))
    _write_mask(gt_dir / "001_mask.png")

    dataset = GLASSDataset(
        data_root=str(data_root),
        split="test",
        cls_names=["bottle"],
        multi_class=False,
        img_size=8,
        resize=8,
        fg=0,
        pipeline=[PackADInputs()],
    )

    assert len(dataset) == 2
    packed = dataset[0]
    assert isinstance(packed["inputs"], torch.Tensor)
    assert packed["inputs"].shape == (3, 8, 8)
    sample = packed["data_samples"]
    assert sample.gt_label == 1
    assert sample.gt_mask.shape == (8, 8)
    assert float(sample.gt_mask.sum().item()) > 0.0


def test_glass_dataset_train_uses_distribution_foreground_fallback(
    tmp_path, monkeypatch
):
    data_root = tmp_path / "mvtec"
    train_dir = data_root / "tile" / "train" / "good"
    train_dir.mkdir(parents=True)
    _write_rgb_image(train_dir / "000.png", (10, 20, 30))

    dtd_root = tmp_path / "dtd" / "images" / "banded"
    dtd_root.mkdir(parents=True)
    _write_rgb_image(dtd_root / "tex.png", (50, 100, 150))

    meta_path = tmp_path / "mvtec_distribution.xlsx"
    _write_distribution_xlsx_with_rows(
        meta_path,
        [{"Class": "mvtec_tile", "Distribution": 0, "Foreground": 0}],
    )

    import baoiad.datasets.glass_dataset as glass_dataset_module

    monkeypatch.setattr(
        glass_dataset_module,
        "generate_glass_perlin_masks",
        lambda **kwargs: (
            np.pad(np.ones((2, 2), dtype=np.float32), ((1, 1), (1, 1))),
            np.pad(np.ones((4, 4), dtype=np.float32), ((2, 2), (2, 2))),
        ),
    )

    dataset = GLASSDataset(
        data_root=str(data_root),
        split="train",
        cls_names=["tile"],
        multi_class=False,
        img_size=8,
        resize=8,
        downsampling=2,
        dtd_path=str(dtd_root.parent.parent),
        fg_mask_root=str(tmp_path / "fg_mask"),
        distribution_meta_path=str(meta_path),
        distribution=0,
        fg=1,
        pipeline=[PackGLASSInputs()],
    )

    packed = dataset[0]
    sample = packed["data_samples"]
    assert sample.mask_s.shape == (4, 4)
    assert sample.aug.shape == (3, 8, 8)

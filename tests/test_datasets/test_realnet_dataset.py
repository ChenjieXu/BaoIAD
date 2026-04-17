"""Tests for the RealNet training dataset."""

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

import baoiad  # noqa: F401

from baoiad.datasets.realnet_dataset import RealNetTrainDataset
from baoiad.datasets.transforms import PackRealNetInputs


def _write_rgb_image(path: Path, rgb) -> None:
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    img[..., 0] = rgb[2]
    img[..., 1] = rgb[1]
    img[..., 2] = rgb[0]
    cv2.imwrite(str(path), img)


def _write_rgb_array(path: Path, rgb_array: np.ndarray) -> None:
    bgr = rgb_array[..., ::-1].copy()
    cv2.imwrite(str(path), bgr)


def test_realnet_dataset_normal_sample(tmp_path, monkeypatch):
    data_root = tmp_path / 'mvtec'
    train_dir = data_root / 'bottle' / 'train' / 'good'
    train_dir.mkdir(parents=True)
    _write_rgb_image(train_dir / '000.png', (10, 20, 30))

    dataset = RealNetTrainDataset(
        data_root=str(data_root),
        cls_names=['bottle'],
        img_size=8,
        dtd_dir=None,
        sdas_dir=None,
        anomaly_types={'normal': 1.0},
        pipeline=[],
    )
    monkeypatch.setattr(dataset, 'choice_anomaly_type', lambda cls_name: 'normal')

    sample = dataset[0]
    assert sample['img'].shape == (3, 8, 8)
    assert sample['gt_img'].shape == (3, 8, 8)
    assert sample['gt_mask'].shape == (8, 8)
    assert float(sample['anomaly_mask'].sum()) == 0.0
    assert sample['anomaly_type'] == 'normal'

    expected = ((10.0 / 255.0) - 0.485) / 0.229
    assert np.isclose(sample['gt_img'][0, 0, 0], expected, atol=1e-5)


def test_realnet_dataset_sdas_sample_and_pack(tmp_path, monkeypatch):
    data_root = tmp_path / 'mvtec'
    train_dir = data_root / 'bottle' / 'train' / 'good'
    train_dir.mkdir(parents=True)
    _write_rgb_image(train_dir / '000.png', (30, 60, 90))

    sdas_dir = data_root / 'sdas' / 'bottle'
    sdas_dir.mkdir(parents=True)
    _write_rgb_image(sdas_dir / 'sdas.png', (200, 100, 50))

    dataset = RealNetTrainDataset(
        data_root=str(data_root),
        cls_names=['bottle'],
        img_size=8,
        dtd_dir=None,
        sdas_dir='auto',
        anomaly_types={'sdas': 1.0},
        pipeline=[PackRealNetInputs()],
    )
    monkeypatch.setattr(dataset, 'choice_anomaly_type', lambda cls_name: 'sdas')
    monkeypatch.setattr(
        dataset,
        'generate_target_foreground_mask',
        lambda image, cls_name: np.ones((8, 8), dtype=np.float32),
    )
    monkeypatch.setattr(
        dataset,
        'generate_perlin_noise_mask',
        lambda: np.pad(np.ones((4, 4), dtype=np.float32), ((2, 2), (2, 2))),
    )

    packed = dataset[0]
    assert isinstance(packed['inputs'], torch.Tensor)
    assert packed['inputs'].shape == (3, 8, 8)
    sample = packed['data_samples']
    assert hasattr(sample, 'clean_img')
    assert hasattr(sample, 'gt_mask')
    assert sample.anomaly_type == 'sdas'
    assert sample.gt_mask.shape == (8, 8)
    assert set(torch.unique(sample.gt_mask).tolist()) <= {0.0, 1.0}
    assert float(sample.gt_mask.sum().item()) > 0.0


def test_realnet_dataset_uses_official_texture_augmenter_pool(tmp_path):
    data_root = tmp_path / 'mvtec'
    train_dir = data_root / 'bottle' / 'train' / 'good'
    train_dir.mkdir(parents=True)
    _write_rgb_image(train_dir / '000.png', (30, 60, 90))

    dataset = RealNetTrainDataset(
        data_root=str(data_root),
        cls_names=['bottle'],
        img_size=8,
        dtd_dir=None,
        sdas_dir=None,
        anomaly_types={'normal': 1.0},
        pipeline=[],
    )

    assert len(dataset._augmenters) == 10

    source = np.full((8, 8, 3), 128, dtype=np.uint8)
    augmented = dataset._rand_augmenter()(image=source)
    assert augmented.shape == source.shape
    assert augmented.dtype == np.uint8


def test_realnet_dataset_clean_image_follows_official_pil_resize(tmp_path, monkeypatch):
    data_root = tmp_path / 'mvtec'
    train_dir = data_root / 'bottle' / 'train' / 'good'
    train_dir.mkdir(parents=True)

    rgb = np.array(
        [
            [[0, 32, 64], [96, 128, 160], [192, 224, 255]],
            [[16, 48, 80], [112, 144, 176], [208, 240, 255]],
            [[8, 24, 40], [56, 72, 88], [104, 120, 136]],
        ],
        dtype=np.uint8,
    )
    _write_rgb_array(train_dir / '000.png', rgb)

    dataset = RealNetTrainDataset(
        data_root=str(data_root),
        cls_names=['bottle'],
        img_size=5,
        dtd_dir=None,
        sdas_dir=None,
        anomaly_types={'normal': 1.0},
        pipeline=[],
    )
    monkeypatch.setattr(dataset, 'choice_anomaly_type', lambda cls_name: 'normal')

    sample = dataset[0]

    pil_resized = np.asarray(Image.fromarray(rgb, mode='RGB').resize((5, 5), resample=Image.BILINEAR))
    expected = pil_resized.astype(np.float32) / 255.0
    expected = ((expected.transpose(2, 0, 1) - dataset.pixel_mean) / dataset.pixel_std)

    assert np.allclose(sample['gt_img'], expected, atol=1e-6)

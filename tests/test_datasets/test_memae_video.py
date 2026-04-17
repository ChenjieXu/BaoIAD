"""Tests for official-style MemAE video dataset."""

from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio

import baoiad  # noqa: F401
from baoiad.registry import DATASETS


def test_memae_official_clip_dataset_reads_processed_layout(tmp_path: Path):
    root = tmp_path / 'UCSD_P2_256'
    train_video = root / 'Train' / 'Train001'
    train_idx = root / 'Train_idx' / 'Train001'
    test_video = root / 'Test' / 'Test001'
    test_idx = root / 'Test_idx' / 'Test001'
    test_gt = root / 'Test_gt'

    for path in [train_video, train_idx, test_video, test_idx, test_gt]:
        path.mkdir(parents=True, exist_ok=True)

    for frame_idx in range(1, 7):
        image = np.full((8, 8), frame_idx * 10, dtype=np.uint8)
        cv2.imwrite(str(train_video / f'{frame_idx:03d}.jpg'), image)
        cv2.imwrite(str(test_video / f'{frame_idx:03d}.jpg'), image)

    sio.savemat(str(train_idx / 'Train001_i001.mat'), {'v_name': 'Train001', 'idx': np.array([[1, 2, 3, 4]])})
    sio.savemat(str(test_idx / 'Test001_i001.mat'), {'v_name': 'Test001', 'idx': np.array([[2, 3, 4, 5]])})
    sio.savemat(str(test_gt / 'Test001.mat'), {'l': np.array([[0, 0, 1, 1, 0, 0]])})

    train_dataset = DATASETS.build(
        dict(
            type='MemAEOfficialClipDataset',
            data_root=str(root),
            split='train',
            dataset_name='UCSDped2',
            clip_length=4,
            in_channels=1,
            img_size=8,
        )
    )
    test_dataset = DATASETS.build(
        dict(
            type='MemAEOfficialClipDataset',
            data_root=str(root),
            split='test',
            dataset_name='UCSDped2',
            clip_length=4,
            in_channels=1,
            img_size=8,
        )
    )

    train_item = train_dataset[0]
    test_item = test_dataset[0]

    assert tuple(train_item['inputs'].shape) == (1, 4, 8, 8)
    assert tuple(test_item['inputs'].shape) == (1, 4, 8, 8)
    assert int(train_item['data_samples'].gt_label) == 0
    # target_frame_offset = clip_length // 2 = 2 -> frame number 4 -> anomaly
    assert int(test_item['data_samples'].gt_label) == 1
    assert test_item['data_samples'].video_name == 'Test001'
    assert int(test_item['data_samples'].frame_idx) == 4

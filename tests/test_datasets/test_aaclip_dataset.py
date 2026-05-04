"""Tests for AA-CLIP jsonl dataset."""

from __future__ import annotations

import json

import numpy as np
from PIL import Image

from baoiad.datasets.aaclip_dataset import AACLIPJsonDataset


def test_aaclip_json_dataset_reads_metadata(tmp_path):
    data_root = tmp_path / 'visa'
    image_dir = data_root / 'candle' / 'train' / 'good'
    mask_dir = data_root / 'candle' / 'ground_truth' / 'bad'
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    image = np.full((8, 8, 3), 127, dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 255
    Image.fromarray(image).save(image_dir / '000.png')
    Image.fromarray(mask).save(mask_dir / '000.png')

    metadata_path = tmp_path / 'visa.jsonl'
    rows = [
        dict(image_path='candle/train/good/000.png', label=0, class_name='candle'),
        dict(
            image_path='candle/train/good/000.png',
            label=1,
            mask_path='candle/ground_truth/bad/000.png',
            class_name='candle',
        ),
    ]
    with metadata_path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row) + '\n')

    dataset = AACLIPJsonDataset(
        data_root=str(data_root),
        metadata_path=str(metadata_path),
        dataset_name='VisA',
        img_size=8,
        multi_class=True,
        augment=False,
    )

    assert len(dataset) == 2
    sample = dataset[1]
    assert tuple(sample['inputs'].shape) == (3, 8, 8)
    assert sample['data_samples'].gt_label == 1
    assert tuple(sample['data_samples'].gt_mask.shape) == (8, 8)


def test_aaclip_json_dataset_remaps_official_visa_paths(tmp_path):
    data_root = tmp_path / 'visa'
    image_dir = data_root / 'candle' / 'test' / 'bad'
    mask_dir = data_root / 'candle' / 'ground_truth' / 'bad'
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    image = np.full((8, 8, 3), 127, dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 255
    Image.fromarray(image).save(image_dir / '000.png')
    Image.fromarray(mask).save(mask_dir / '000.png')

    metadata_path = tmp_path / 'visa_official.jsonl'
    rows = [
        dict(
            image_path='candle/Data/Images/Anomaly/000.png',
            label=1,
            mask_path='candle/Data/Masks/Anomaly/000.png',
            class_name='candle',
        ),
    ]
    with metadata_path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row) + '\n')

    dataset = AACLIPJsonDataset(
        data_root=str(data_root),
        metadata_path=str(metadata_path),
        dataset_name='VisA',
        img_size=8,
        multi_class=True,
        augment=False,
    )
    dataset.full_init()

    data_info = dataset.get_data_info(0)
    assert data_info['img_path'].endswith('candle/test/bad/000.png')
    assert data_info['gt_mask_path'].endswith('candle/ground_truth/bad/000.png')
    sample = dataset[0]
    assert sample['data_samples'].gt_label == 1
    assert tuple(sample['data_samples'].gt_mask.shape) == (8, 8)

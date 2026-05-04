"""Tests for strict CFlow preprocessing transforms."""

import numpy as np

import baoiad  # noqa: F401
from baoiad.registry import TRANSFORMS


def test_cflow_official_transform_applies_category_size_and_binary_mask():
    transform = TRANSFORMS.build(
        dict(
            type='CFlowOfficialTransform',
            size_map={'bottle': 32, 'transistor': 16},
            default_size=24,
            train=False,
        )
    )

    img = np.zeros((40, 80, 3), dtype=np.uint8)
    mask = np.zeros((40, 80), dtype=np.float32)
    mask[10:20, 20:30] = 1.0

    results = transform(
        dict(
            img=img,
            gt_mask=mask,
            cls_name='bottle',
        )
    )

    assert results['img'].shape == (32, 32, 3)
    assert results['gt_mask'].shape == (32, 32)
    assert set(np.unique(results['gt_mask']).tolist()).issubset({0.0, 1.0})


def test_cflow_official_transform_train_path_keeps_square_shape():
    transform = TRANSFORMS.build(
        dict(
            type='CFlowOfficialTransform',
            size_map={'transistor': 16},
            default_size=24,
            train=True,
        )
    )

    img = np.full((24, 48, 3), 127, dtype=np.uint8)
    results = transform(dict(img=img, cls_name='transistor'))

    assert results['img'].shape == (16, 16, 3)
    assert results['img'].dtype == np.float32

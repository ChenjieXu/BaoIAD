"""Shared fixtures for BaoIAD tests."""

import numpy as np
import pytest
import torch

from baoiad import register_all_modules
from baoiad.structures import ADDataSample

register_all_modules()


@pytest.fixture
def dummy_image():
    """Return a (3, 256, 256) float tensor."""
    return torch.randn(3, 256, 256)


@pytest.fixture
def dummy_data_samples():
    """Return a list of 4 ADDataSample with gt_label, gt_mask, cls_name."""
    samples = []
    for i in range(4):
        s = ADDataSample()
        s.gt_label = i % 2  # alternating normal / anomalous
        s.gt_mask = torch.zeros(256, 256) if i % 2 == 0 else torch.ones(256, 256)
        s.cls_name = "bottle"
        s.img_path = f"/fake/img_{i}.png"
        s.defect_type = "good" if i % 2 == 0 else "broken"
        samples.append(s)
    return samples


@pytest.fixture
def tmp_mvtec_dir(tmp_path):
    """Create a minimal MVTec-like directory with tiny images and masks."""
    cls_name = "bottle"
    # train/good
    train_good = tmp_path / cls_name / "train" / "good"
    train_good.mkdir(parents=True)
    # test/good and test/broken_large
    test_good = tmp_path / cls_name / "test" / "good"
    test_good.mkdir(parents=True)
    test_defect = tmp_path / cls_name / "test" / "broken_large"
    test_defect.mkdir(parents=True)
    # ground_truth/broken_large
    gt_dir = tmp_path / cls_name / "ground_truth" / "broken_large"
    gt_dir.mkdir(parents=True)

    import cv2

    small_img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    mask_img = np.zeros((32, 32), dtype=np.uint8)
    mask_img[8:24, 8:24] = 255

    for i in range(3):
        cv2.imwrite(str(train_good / f"{i:03d}.png"), small_img)
        cv2.imwrite(str(test_good / f"{i:03d}.png"), small_img)
        cv2.imwrite(str(test_defect / f"{i:03d}.png"), small_img)
        cv2.imwrite(str(gt_dir / f"{i:03d}_mask.png"), mask_img)

    return tmp_path

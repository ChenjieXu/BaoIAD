"""Tests for VisADataset."""

import cv2
import numpy as np

import baoiad  # noqa: F401

from baoiad.datasets.visa import VisADataset


class TestVisADataset:
    def test_load_data_list_mvt_like_fallback(self, tmp_path):
        cls_name = 'candle'
        train_good = tmp_path / cls_name / 'train' / 'good'
        test_good = tmp_path / cls_name / 'test' / 'good'
        test_bad = tmp_path / cls_name / 'test' / 'bad'
        gt_bad = tmp_path / cls_name / 'ground_truth' / 'bad'
        train_good.mkdir(parents=True)
        test_good.mkdir(parents=True)
        test_bad.mkdir(parents=True)
        gt_bad.mkdir(parents=True)

        img = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[4:12, 4:12] = 255

        cv2.imwrite(str(train_good / '000.JPG'), img)
        cv2.imwrite(str(test_good / '001.JPG'), img)
        cv2.imwrite(str(test_bad / '002.JPG'), img)
        cv2.imwrite(str(gt_bad / '002.png'), mask)

        train_ds = VisADataset(
            data_root=str(tmp_path),
            split='train',
            cls_names=[cls_name],
            multi_class=False,
        )
        test_ds = VisADataset(
            data_root=str(tmp_path),
            split='test',
            cls_names=[cls_name],
            multi_class=False,
        )

        train_data = train_ds.load_data_list()
        test_data = test_ds.load_data_list()

        assert len(train_data) == 1
        assert train_data[0]['gt_label'] == 0
        assert train_data[0]['gt_mask_path'] == ''

        assert len(test_data) == 2
        anomalous = [item for item in test_data if item['gt_label'] == 1]
        assert len(anomalous) == 1
        assert anomalous[0]['gt_mask_path'].endswith('002.png')

"""Tests for AdaCLIP strict auxiliary datasets."""

import json

import cv2
import numpy as np

import baoiad  # noqa: F401

from baoiad.datasets.adaclip_aux import AdaCLIPClinicDBDataset, AdaCLIPVisADataset


def _write_rgb(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
    cv2.imwrite(str(path), image)


def _write_mask(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 4:12] = 255
    cv2.imwrite(str(path), mask)


class TestAdaCLIPStrictDatasets:
    def test_adaclip_visa_prefers_meta_test_partition(self, tmp_path):
        _write_rgb(tmp_path / 'candle' / 'train' / 'good' / 'train_only.JPG')
        _write_rgb(tmp_path / 'candle' / 'test' / 'bad' / 'test_only.JPG')
        _write_mask(tmp_path / 'candle' / 'ground_truth' / 'bad' / 'test_only.png')

        meta = {
            'train': {
                'candle': [
                    {
                        'img_path': 'candle/train/good/train_only.JPG',
                        'mask_path': '',
                        'cls_name': 'candle',
                        'specie_name': 'good',
                        'anomaly': 0,
                    }
                ]
            },
            'test': {
                'candle': [
                    {
                        'img_path': 'candle/test/bad/test_only.JPG',
                        'mask_path': 'candle/ground_truth/bad/test_only.png',
                        'cls_name': 'candle',
                        'specie_name': 'bad',
                        'anomaly': 1,
                    }
                ]
            },
        }
        (tmp_path / 'meta.json').write_text(json.dumps(meta), encoding='utf-8')

        dataset = AdaCLIPVisADataset(
            data_root=str(tmp_path),
            split='train',
            cls_names=['candle'],
            multi_class=False,
        )
        data_list = dataset.load_data_list()

        assert len(data_list) == 1
        assert data_list[0]['img_path'].endswith('candle/test/bad/test_only.JPG')
        assert data_list[0]['gt_label'] == 1
        assert data_list[0]['defect_type'] == 'bad'

    def test_adaclip_clinicdb_fallback_uses_test_layout(self, tmp_path):
        _write_rgb(tmp_path / 'ClinicDB' / 'train' / 'polyp' / 'train_only.png')
        _write_rgb(tmp_path / 'ClinicDB' / 'test' / 'polyp' / 'test_only.png')
        _write_mask(tmp_path / 'ClinicDB' / 'ground_truth' / 'polyp' / 'test_only.png')

        dataset = AdaCLIPClinicDBDataset(
            data_root=str(tmp_path),
            split='train',
            cls_names=['ClinicDB'],
            multi_class=False,
        )
        data_list = dataset.load_data_list()

        assert len(data_list) == 1
        assert data_list[0]['img_path'].endswith('ClinicDB/test/polyp/test_only.png')
        assert data_list[0]['gt_mask_path'].endswith('ClinicDB/ground_truth/polyp/test_only.png')
        assert data_list[0]['gt_label'] == 1
        assert data_list[0]['defect_type'] == 'polyp'

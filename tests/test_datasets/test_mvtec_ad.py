"""Tests for MVTecADDataset."""

import random

import baoiad  # noqa: F401

from baoiad.datasets.mvtec_ad import MVTecADDataset


class TestMVTecADDataset:
    def test_load_data_list_train(self, tmp_mvtec_dir):
        ds = MVTecADDataset(
            data_root=str(tmp_mvtec_dir),
            split='train',
            cls_names=['bottle'],
            multi_class=False,
        )
        data_list = ds.load_data_list()
        assert len(data_list) == 3
        assert all(d['gt_label'] == 0 for d in data_list)

    def test_load_data_list_test(self, tmp_mvtec_dir):
        ds = MVTecADDataset(
            data_root=str(tmp_mvtec_dir),
            split='test',
            cls_names=['bottle'],
            multi_class=False,
        )
        data_list = ds.load_data_list()
        # 3 good + 3 broken_large
        assert len(data_list) == 6
        normal = [d for d in data_list if d['gt_label'] == 0]
        anomalous = [d for d in data_list if d['gt_label'] == 1]
        assert len(normal) == 3
        assert len(anomalous) == 3
        # Anomalous should have mask paths
        assert all(d['gt_mask_path'] != '' for d in anomalous)

    def test_all_categories_count(self):
        assert len(MVTecADDataset.ALL_CATEGORIES) == 15

    def test_shuffle_train_data_changes_order_deterministically(self, tmp_mvtec_dir):
        random.seed(42)
        shuffled = MVTecADDataset(
            data_root=str(tmp_mvtec_dir),
            split='train',
            cls_names=['bottle'],
            multi_class=False,
            shuffle_train_data=True,
        ).load_data_list()

        baseline = MVTecADDataset(
            data_root=str(tmp_mvtec_dir),
            split='train',
            cls_names=['bottle'],
            multi_class=False,
            shuffle_train_data=False,
        ).load_data_list()

        assert len(shuffled) == len(baseline)
        assert [item['img_path'] for item in shuffled] != [item['img_path'] for item in baseline]

        random.seed(42)
        reshuffled = MVTecADDataset(
            data_root=str(tmp_mvtec_dir),
            split='train',
            cls_names=['bottle'],
            multi_class=False,
            shuffle_train_data=True,
        ).load_data_list()
        assert [item['img_path'] for item in reshuffled] == [item['img_path'] for item in shuffled]

    def test_train_val_split_exposes_deterministic_official_subsets(self, tmp_mvtec_dir):
        train_subset = MVTecADDataset(
            data_root=str(tmp_mvtec_dir),
            split='train',
            cls_names=['bottle'],
            multi_class=False,
            train_val_split_ratio=0.2,
            train_val_split_seed=0,
            train_val_split_subset='train',
        ).load_data_list()
        val_subset = MVTecADDataset(
            data_root=str(tmp_mvtec_dir),
            split='train',
            cls_names=['bottle'],
            multi_class=False,
            train_val_split_ratio=0.2,
            train_val_split_seed=0,
            train_val_split_subset='val',
        ).load_data_list()

        assert len(train_subset) == 2
        assert len(val_subset) == 1
        assert {
            item['img_path'] for item in train_subset
        }.isdisjoint({item['img_path'] for item in val_subset})

        train_subset_again = MVTecADDataset(
            data_root=str(tmp_mvtec_dir),
            split='train',
            cls_names=['bottle'],
            multi_class=False,
            train_val_split_ratio=0.2,
            train_val_split_seed=0,
            train_val_split_subset='train',
        ).load_data_list()
        assert [item['img_path'] for item in train_subset_again] == [
            item['img_path'] for item in train_subset
        ]

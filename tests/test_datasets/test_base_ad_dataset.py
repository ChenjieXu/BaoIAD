"""Tests for BaseADDataset."""

import pytest

import baoiad  # noqa: F401

from baoiad.datasets.base_ad_dataset import BaseADDataset


class _DummyADDataset(BaseADDataset):
    ALL_CATEGORIES = ('cat_a', 'cat_b', 'cat_c')

    def load_data_list(self):
        return [{'img_path': f'/fake/{c}/{i}.png', 'gt_label': 0, 'cls_name': c}
                for c in self.cls_names for i in range(2)]


class TestBaseADDataset:
    def test_multi_class_default(self, tmp_path):
        ds = _DummyADDataset(data_root=str(tmp_path), multi_class=True)
        assert set(ds.cls_names) == {'cat_a', 'cat_b', 'cat_c'}

    def test_single_class_requires_cls_names(self, tmp_path):
        with pytest.raises(ValueError, match='cls_names must be provided'):
            _DummyADDataset(data_root=str(tmp_path), multi_class=False)

    def test_single_class_with_cls_names(self, tmp_path):
        ds = _DummyADDataset(data_root=str(tmp_path), multi_class=False, cls_names=['cat_a'])
        assert ds.cls_names == ['cat_a']

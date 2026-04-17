"""Tests for MultiScalePooling neck."""

import pytest
import torch

import baoiad  # noqa: F401

from baoiad.models.necks.multi_scale_pooling import MultiScalePooling


class TestMultiScalePooling:
    @pytest.mark.parametrize('output_size', [14, 28])
    def test_forward(self, output_size):
        neck = MultiScalePooling(output_size=output_size)
        feats = (torch.randn(2, 64, 56, 56), torch.randn(2, 128, 28, 28))
        out = neck(feats)
        assert len(out) == 2
        for f in out:
            assert f.shape[-2:] == (output_size, output_size)

    def test_already_correct_size(self):
        neck = MultiScalePooling(output_size=28)
        feat = (torch.randn(2, 64, 28, 28),)
        out = neck(feat)
        assert out[0].shape == (2, 64, 28, 28)

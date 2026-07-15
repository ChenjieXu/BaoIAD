"""Tests for UFlowDetector."""

import importlib
import os
import tempfile
from unittest import TestCase
from unittest.mock import patch

import pytest
import torch

pytestmark = pytest.mark.optional
pytest.importorskip("FrEIA", reason='requires the "flow" optional extra')

importlib.import_module("baoiad")
MODELS = importlib.import_module("baoiad.registry").MODELS
ADDataSample = importlib.import_module("baoiad.structures").ADDataSample
resolve_cait_pretrained_overlays = importlib.import_module(
    "baoiad.models.detectors.uflow"
).resolve_cait_pretrained_overlays


def _make_data_samples(batch_size, H=256, W=256):
    samples = []
    for i in range(batch_size):
        s = ADDataSample()
        s.gt_label = i % 2
        s.gt_mask = torch.zeros(H, W)
        s.cls_name = "bottle"
        s.img_path = f"/fake/{i}.png"
        s.defect_type = "good"
        samples.append(s)
    return samples


class TestUFlowDetector(TestCase):
    def setUp(self):
        self.cfg = dict(
            type="UFlowDetector", input_size=(64, 64), flow_steps=2, backbone="resnet18"
        )

    def test_forward_tensor(self):
        model = MODELS.build(self.cfg)
        model.eval()
        out = model(torch.randn(2, 3, 64, 64), mode="tensor")
        assert out is not None

    def test_forward_loss(self):
        model = MODELS.build(self.cfg)
        model.train()
        data_samples = _make_data_samples(2, 64, 64)
        out = model(torch.randn(2, 3, 64, 64), data_samples, mode="loss")
        assert isinstance(out, dict)

    def test_forward_predict(self):
        model = MODELS.build(self.cfg)
        model.eval()
        data_samples = _make_data_samples(2, 64, 64)
        out = model(torch.randn(2, 3, 64, 64), data_samples, mode="predict")
        assert isinstance(out, list)
        assert len(out) == 2
        assert hasattr(out[0], "pred_anomaly_map")
        assert tuple(out[0].pred_anomaly_map.shape) == (1, 64, 64)
        assert torch.isfinite(out[0].pred_anomaly_map).all()
        assert not hasattr(out[0], "pred_nfa_anomaly_map")
        assert not hasattr(out[0], "pred_nfa_score")
        assert out[0].pred_score == pytest.approx(float(out[0].pred_anomaly_map.max()))

    def test_forward_predict_can_attach_nfa_outputs(self):
        model = MODELS.build(dict(**self.cfg, compute_nfa_in_predict=True))
        model.eval()
        data_samples = _make_data_samples(2, 64, 64)
        fake_nfa = torch.full((2, 1, 64, 64), 0.25)
        with patch.object(model, "compute_nfa_anomaly_map", return_value=fake_nfa):
            out = model(torch.randn(2, 3, 64, 64), data_samples, mode="predict")
        assert hasattr(out[0], "pred_nfa_anomaly_map")
        assert hasattr(out[0], "pred_nfa_score")
        assert tuple(out[0].pred_nfa_anomaly_map.shape) == (1, 64, 64)
        assert torch.isfinite(out[0].pred_nfa_anomaly_map).all()
        assert out[0].pred_nfa_score == pytest.approx(0.25)


def test_resolve_cait_pretrained_overlays_prefers_local_cached_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        m48 = os.path.join(tmpdir, "M48_448.pth")
        s24 = os.path.join(tmpdir, "S24_224.pth")
        open(m48, "wb").close()
        open(s24, "wb").close()

        overlays = resolve_cait_pretrained_overlays(tmpdir)

    assert overlays["cait_m48_448"]["file"] == m48
    assert overlays["cait_m48_448"]["hf_hub_id"] is None
    assert overlays["cait_m48_448"]["source"] == "file"
    assert overlays["cait_s24_224"]["file"] == s24
    assert overlays["cait_s24_224"]["hf_hub_id"] is None
    assert overlays["cait_s24_224"]["source"] == "file"

"""Tests for AnomalyCLIPDetector."""

import os
import sys
from types import SimpleNamespace

import pytest
import torch
from unittest import TestCase
from baoiad.checkpoint import CheckpointLoadError
from baoiad.models.detectors.anomalyclip import AnomalyCLIPDetector
from baoiad.structures import ADDataSample

import baoiad  # noqa: F401
from baoiad.registry import MODELS


def _make_data_samples(batch_size, H=256, W=256, classes=None):
    samples = []
    if classes is None:
        classes = ['bottle'] * batch_size
    for i in range(batch_size):
        s = ADDataSample()
        s.gt_label = i % 2
        s.gt_mask = torch.zeros(H, W)
        s.cls_name = classes[i]
        s.img_path = f'/fake/{i}.png'
        s.defect_type = 'good'
        samples.append(s)
    return samples


try:
    import open_clip
    HAS_OPEN_CLIP = True
except ImportError:
    HAS_OPEN_CLIP = False

@pytest.mark.skipif(not HAS_OPEN_CLIP, reason='open_clip not installed')
class TestAnomalyCLIPDetector(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = dict(
            type='AnomalyCLIPDetector',
            clip_model='ViT-B-32',
            pretrained=None,
            n_prompt_tokens=2,
            image_size=32,
            features_list=[3, 6, 9, 12],
            dpam_enabled=False,
            use_view_augmentation=False,
            official_checkpoint=None,
        )
        cls.model = MODELS.build(cls.cfg)

    def setUp(self):
        self.model = self.__class__.model

    def test_forward_tensor(self):
        self.model.eval()
        out = self.model(torch.randn(2, 3, 32, 32), mode='tensor')
        assert out is not None

    def test_forward_loss(self):
        self.model.train()
        data_samples = _make_data_samples(2, 32, 32)
        out = self.model(torch.randn(2, 3, 32, 32), data_samples, mode='loss')
        assert isinstance(out, dict)

    def test_forward_predict(self):
        self.model.eval()
        data_samples = _make_data_samples(2, 32, 32, classes=['metal_nut', 'bottle'])
        out = self.model(torch.randn(2, 3, 32, 32), data_samples, mode='predict')
        assert isinstance(out, list)
        assert len(out) == 2


def test_official_detector_requires_reference_assets():
    cfg = dict(
        type='AnomalyCLIPOfficialDetector',
        reference_root=os.path.join('/tmp', 'missing_anomalyclip_reference'),
        require_official_assets=True,
    )
    with pytest.raises(FileNotFoundError):
        MODELS.build(cfg)


def test_official_detector_requires_checkpoint_when_configured():
    cfg = dict(
        type='AnomalyCLIPOfficialDetector',
        reference_root='.refs/AnomalyCLIP',
        official_checkpoint=os.path.join('/tmp', 'missing_anomalyclip_epoch_15.pth'),
        require_official_assets=True,
    )
    with pytest.raises(FileNotFoundError):
        MODELS.build(cfg)


def test_official_prompt_loader_does_not_swallow_checkpoint_errors(monkeypatch):
    expected = CheckpointLoadError('checkpoint is corrupt')
    monkeypatch.setattr(sys, 'path', list(sys.path))
    monkeypatch.setitem(
        sys.modules,
        'prompt_ensemble',
        SimpleNamespace(AnomalyCLIP_PromptLearner=object),
    )
    monkeypatch.setattr(
        'baoiad.models.detectors.anomalyclip.load_baoiad_checkpoint',
        lambda *args, **kwargs: (_ for _ in ()).throw(expected),
    )
    detector = object.__new__(AnomalyCLIPDetector)

    with pytest.raises(CheckpointLoadError) as exc_info:
        detector._init_official_prompt_learner('/tmp/corrupt.pth')

    assert exc_info.value is expected

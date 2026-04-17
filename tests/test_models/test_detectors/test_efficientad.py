"""Tests for EfficientADDetector."""

import os
import tempfile
from unittest import TestCase

import torch
from PIL import Image

import baoiad  # noqa: F401
from baoiad.registry import MODELS
from baoiad.structures import ADDataSample


def _make_data_samples(batch_size, H=256, W=256, gt_label_fn=None):
    samples = []
    for i in range(batch_size):
        sample = ADDataSample()
        sample.gt_label = gt_label_fn(i) if gt_label_fn is not None else i % 2
        sample.gt_mask = torch.zeros(H, W)
        sample.set_metainfo({
            'cls_name': 'bottle',
            'img_path': f'/fake/{i}.png',
            'defect_type': 'good',
        })
        samples.append(sample)
    return samples


class TestEfficientADDetector(TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        train_dir = os.path.join(self.temp_dir.name, 'train', 'class0')
        os.makedirs(train_dir, exist_ok=True)
        Image.new('RGB', (256, 256), color=(128, 128, 128)).save(os.path.join(train_dir, 'sample.png'))

        # EfficientAD PDN requires >= 256x256 input due to conv strides
        self.cfg = dict(
            type='EfficientADDetector',
            pdn_channels=64,
            pdn_variant='small',
            padding=False,
            ae_weight=1.0,
            penalty_weight=1.0,
            teacher_pretrained='',
            imagenet_dir=self.temp_dir.name,
        )
        self.H = self.W = 256

    def tearDown(self):
        self.temp_dir.cleanup()

    def _make_loader(self, batch_size=2, num_batches=2, *, good_only=False):
        gt_label_fn = (lambda _: 0) if good_only else None
        return [
            {
                'inputs': torch.rand(batch_size, 3, self.H, self.W),
                'data_samples': _make_data_samples(batch_size, self.H, self.W, gt_label_fn=gt_label_fn),
            }
            for _ in range(num_batches)
        ]

    def test_forward_tensor(self):
        model = MODELS.build(self.cfg)
        model.eval()
        out = model(torch.rand(2, 3, self.H, self.W), mode='tensor')
        assert out is not None

    def test_forward_loss(self):
        model = MODELS.build(self.cfg)
        model.train()
        data_samples = _make_data_samples(2, self.H, self.W)
        out = model(torch.rand(2, 3, self.H, self.W), data_samples, mode='loss')
        assert isinstance(out, dict)
        assert 'loss' in out
        assert torch.isfinite(out['loss'])

    def test_forward_predict(self):
        model = MODELS.build(self.cfg)
        model.eval()
        data_samples = _make_data_samples(2, self.H, self.W)
        out = model(torch.rand(2, 3, self.H, self.W), data_samples, mode='predict')
        assert isinstance(out, list)
        assert len(out) == 2

    def test_pre_train_setup_populates_teacher_stats(self):
        model = MODELS.build(self.cfg)
        loader = self._make_loader()

        model.pre_train_setup(loader)

        assert model._is_mean_std_set()
        assert model._imagenet_loader is not None

    def test_compute_normalization_stats_populates_quantiles(self):
        model = MODELS.build(self.cfg)
        train_loader = self._make_loader()
        val_loader = self._make_loader(good_only=True)

        model.pre_train_setup(train_loader)
        model.compute_normalization_stats(val_loader)

        assert model._is_quantiles_set()
        assert float(model.qb_st.item()) > float(model.qa_st.item())
        assert float(model.qb_ae.item()) > float(model.qa_ae.item())

    def test_predict_after_setup_returns_finite_outputs(self):
        model = MODELS.build(self.cfg)
        train_loader = self._make_loader()
        val_loader = self._make_loader(good_only=True)
        data_samples = _make_data_samples(2, self.H, self.W)

        model.pre_train_setup(train_loader)
        model.compute_normalization_stats(val_loader)
        outputs = model(torch.rand(2, 3, self.H, self.W), data_samples, mode='predict')

        scores = torch.tensor([sample.pred_score for sample in outputs], dtype=torch.float32)
        maps = torch.stack([sample.pred_anomaly_map for sample in outputs])
        assert torch.isfinite(scores).all()
        assert torch.isfinite(maps).all()

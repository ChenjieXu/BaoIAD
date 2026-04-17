"""Tests for ResADDetector."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

pytest.importorskip('FrEIA')  # Skip entire module if FrEIA not installed

import baoiad  # noqa: F401
from baoiad.registry import MODELS
from baoiad.structures import ADDataSample


def _make_data_samples(batch_size, H=64, W=64, cls_names='bottle'):
    if isinstance(cls_names, str):
        cls_names = [cls_names] * batch_size

    samples = []
    for i in range(batch_size):
        sample = ADDataSample()
        sample.gt_label = i % 2
        sample.gt_mask = torch.zeros(H, W)
        sample.cls_name = cls_names[i]
        sample.img_path = f'/fake/{i}.png'
        sample.defect_type = 'good'
        samples.append(sample)
    return samples


def _write_fake_rgb(path: Path, size=(64, 64)):
    array = np.random.randint(0, 255, size + (3,), dtype=np.uint8)
    Image.fromarray(array).save(path)


class MockDataLoader:
    def __init__(self, data, cls_names=None):
        self.data = data
        self.dataset = SimpleNamespace(cls_names=cls_names)

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)


class TestResADDetector:
    def setup_method(self):
        self.cfg = dict(
            type='ResADDetector',
            backbone=dict(
                type='TIMMBackbone',
                model_name='resnet18',
                features_only=True,
                out_indices=(1, 2, 3),
                pretrained=False,
                frozen=True,
            ),
            n_shot=2,
            num_embeddings=64,
            pos_embed_dim=64,
            coupling_layers=2,
            input_size=64,
        )

    def test_forward_tensor(self):
        model = MODELS.build(self.cfg)
        model.eval()
        out = model(torch.randn(2, 3, 64, 64), mode='tensor')
        assert out is not None
        assert len(out) == 3

    def test_forward_loss(self):
        model = MODELS.build(self.cfg)
        model.train()
        data_samples = _make_data_samples(4, 64, 64)
        out = model(torch.randn(4, 3, 64, 64), data_samples, mode='loss')
        assert isinstance(out, dict)
        assert 'loss' in out
        assert out['loss'].item() >= 0

    def test_forward_loss_uses_dataset_references_when_available(self, tmp_path):
        good_dir = tmp_path / 'bottle' / 'train' / 'good'
        good_dir.mkdir(parents=True)
        for index in range(3):
            _write_fake_rgb(good_dir / f'{index:03d}.png')

        cfg = dict(self.cfg)
        cfg.update(data_root=str(tmp_path))
        model = MODELS.build(cfg)
        model.train()

        with patch.object(model, '_sample_reference_from_batch', side_effect=AssertionError):
            out = model(torch.randn(2, 3, 64, 64), _make_data_samples(2, 64, 64), mode='loss')

        assert out['loss'].item() >= 0

    def test_forward_predict_with_memory_bank(self):
        model = MODELS.build(self.cfg)
        model.eval()

        mock_data = [
            {'inputs': torch.randn(2, 3, 64, 64), 'data_samples': _make_data_samples(2, 64, 64, 'bottle')},
            {'inputs': torch.randn(2, 3, 64, 64), 'data_samples': _make_data_samples(2, 64, 64, 'capsule')},
        ]
        model.build_memory_bank(MockDataLoader(mock_data, cls_names=['bottle', 'capsule']))

        data_samples = _make_data_samples(2, 64, 64, ['bottle', 'capsule'])
        out = model(torch.randn(2, 3, 64, 64), data_samples, mode='predict')
        assert isinstance(out, list)
        assert len(out) == 2
        assert hasattr(out[0], 'pred_score')
        assert hasattr(out[0], 'pred_anomaly_map')
        assert torch.isfinite(out[0].pred_anomaly_map).all()

    def test_build_memory_bank_loads_reference_feature_dir(self, tmp_path):
        model = MODELS.build(self.cfg)
        model.eval()

        channels = list(model.backbone.out_channels)
        ref_root = tmp_path / 'refs'
        for cls_name in ['bottle', 'capsule']:
            cls_dir = ref_root / cls_name
            cls_dir.mkdir(parents=True)
            for index, channel in enumerate(channels, start=1):
                np.save(cls_dir / f'layer{index}.npy', np.random.randn(12, channel).astype(np.float32))

        model.ref_feature_dir = str(ref_root)
        model.num_ref_shot = 2
        model.total_ref_shot = 4
        model.build_memory_bank(MockDataLoader([], cls_names=['bottle', 'capsule']))

        assert isinstance(model.ref_bank, dict)
        assert set(model.ref_bank.keys()) == {'bottle', 'capsule'}
        assert all(item.shape[0] > 0 for item in model.ref_bank['bottle'])

    def test_build_memory_bank_respects_strict_reference_requirement(self):
        cfg = dict(self.cfg)
        cfg.update(ref_feature_dir='/nonexistent/resad_refs', strict_ref_features=True)
        model = MODELS.build(cfg)
        with pytest.raises(RuntimeError, match='strict reference mode'):
            model.build_memory_bank(MockDataLoader([], cls_names=['bottle']))

    def test_score_all_uses_dataset_global_max(self):
        model = MODELS.build(self.cfg)
        model.eval()
        model.smooth_sigma = 0

        model._pending_output_size = (1, 1)
        model._pending_logp_batches = [
            [
                torch.tensor([[[0.0]]]),
                torch.tensor([[[0.0]]]),
                torch.tensor([[[0.0]]]),
            ],
            [
                torch.tensor([[[2.0]]]),
                torch.tensor([[[0.0]]]),
                torch.tensor([[[0.0]]]),
            ],
        ]
        neg_large = torch.tensor([[[-100.0]]])
        model._pending_logp_a_batches = [
            [neg_large.clone(), neg_large.clone(), neg_large.clone()],
            [neg_large.clone(), neg_large.clone(), neg_large.clone()],
        ]
        model._pending_samples = [ADDataSample(), ADDataSample()]

        results = model.score_all()

        assert len(results) == 2
        assert results[0].pred_score > 0.4
        assert results[1].pred_score == pytest.approx(0.0, abs=1e-6)
        assert results[0].pred_score > results[1].pred_score
        assert tuple(results[0].pred_anomaly_map.shape) == (1, 1, 1)
        assert model._pending_samples == []
        assert model._pending_logp_batches == []
        assert model._pending_logp_a_batches == []

    def test_flow_loss_stage1_uses_logsigmoid(self):
        model = MODELS.build(self.cfg)

        logps = torch.tensor([0.5, -0.25, 1.0], dtype=torch.float32)
        mask = torch.zeros(3, dtype=torch.long)

        loss = model._compute_flow_loss_stage1(logps, mask)
        expected = -torch.nn.functional.logsigmoid(logps).mean()

        assert torch.allclose(loss, expected)

    def test_flow_loss_stage2_uses_ml_only_for_all_normal_batch(self):
        model = MODELS.build(self.cfg)

        logps = torch.tensor([0.5, -0.25, 1.0], dtype=torch.float32)
        logps_a = torch.tensor([-0.5, -0.75, -0.1], dtype=torch.float32)
        mask = torch.zeros(3, dtype=torch.long)

        loss = model._compute_flow_loss_stage2(logps, logps_a, mask, (0.0, -0.1))
        expected = -torch.nn.functional.logsigmoid(logps).mean()

        assert torch.allclose(loss, expected)

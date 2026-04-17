"""Tests for MemSegDetector."""

import numpy as np
import pytest
import torch

import baoiad  # noqa: F401
from baoiad.registry import MODELS
from baoiad.structures import ADDataSample


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


def _make_cfg(**overrides):
    cfg = dict(
        type='MemSegDetector',
        backbone=dict(
            type='TIMMBackbone',
            model_name='resnet18',
            pretrained=False,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
            frozen=True,
        ),
        nb_memory_sample=3,
        dtd_path='',
        anomaly_ratio=0.5,
        alternate_anomaly_sampling=False,
        require_texture_source=False,
        memory_bank_seed=42,
        anomaly_source_resize=None,
        anomaly_source_crop=None,
    )
    cfg.update(overrides)
    return cfg


def _make_inputs(batch_size=2, height=64, width=64):
    raw = torch.rand(batch_size, 3, height, width, dtype=torch.float32)
    return (raw - _IMAGENET_MEAN) / _IMAGENET_STD


def _make_data_samples(batch_size, cls_name='grid', height=64, width=64):
    samples = []
    for idx in range(batch_size):
        sample = ADDataSample()
        sample.gt_label = idx % 2
        sample.gt_mask = torch.zeros(height, width, dtype=torch.float32)
        sample.set_metainfo({
            'cls_name': cls_name,
            'img_path': f'/fake/{idx}.png',
            'defect_type': 'good',
        })
        samples.append(sample)
    return samples


class _IndexedDataset:
    def __init__(self, num_items=6, height=64, width=64):
        self.accessed = []
        self.samples = []
        for idx in range(num_items):
            raw = torch.full((1, 3, height, width), float(idx + 1) / float(num_items + 1))
            self.samples.append((raw - _IMAGENET_MEAN) / _IMAGENET_STD)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        self.accessed.append(int(idx))
        return {'inputs': self.samples[int(idx)].squeeze(0)}


class _LoaderWithDataset:
    def __init__(self, dataset):
        self.dataset = dataset


def test_build_memory_bank_uses_seeded_dataset_sampling_once():
    model = MODELS.build(_make_cfg(nb_memory_sample=3, memory_bank_seed=42))
    dataset = _IndexedDataset(num_items=6)
    loader = _LoaderWithDataset(dataset)

    model.build_memory_bank(loader)

    expected = np.arange(len(dataset))
    np.random.RandomState(42).shuffle(expected)
    expected = expected[:3].tolist()

    assert dataset.accessed == expected
    assert model.memory_bank is not None
    assert model.memory_bank.nb_memory_sample == 3
    assert model.memory_bank.memory_information['level0'].shape[0] == 3

    model.build_memory_bank(loader)
    assert dataset.accessed == expected


def test_partial_freeze_memory_bank_features_do_not_retain_grad_graph():
    model = MODELS.build(_make_cfg(
        nb_memory_sample=3,
        backbone=dict(
            type='TIMMBackbone',
            model_name='resnet18',
            pretrained=False,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
            frozen=False,
            frozen_names=('layer1', 'layer2', 'layer3'),
        ),
    ))
    dataset = _IndexedDataset(num_items=5)
    loader = _LoaderWithDataset(dataset)

    model.build_memory_bank(loader)

    assert model.memory_bank is not None
    for tensor in model.memory_bank.memory_information.values():
        assert tensor.requires_grad is False


def test_partial_backbone_freeze_matches_reference_trainability():
    model = MODELS.build(_make_cfg(
        backbone=dict(
            type='TIMMBackbone',
            model_name='resnet18',
            pretrained=False,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
            frozen=False,
            frozen_names=('layer1', 'layer2', 'layer3'),
            frozen_names_eval=False,
        ),
    ))

    assert model.freeze_backbone is False

    backbone = model.backbone.net
    assert not any(param.requires_grad for param in backbone.layer1.parameters())
    assert not any(param.requires_grad for param in backbone.layer2.parameters())
    assert not any(param.requires_grad for param in backbone.layer3.parameters())
    assert any(param.requires_grad for param in backbone.layer4.parameters())
    model.train(True)
    assert backbone.layer1.training is True
    assert backbone.layer2.training is True
    assert backbone.layer3.training is True


def test_forward_loss_uses_official_alternating_sampling(monkeypatch):
    model = MODELS.build(_make_cfg(alternate_anomaly_sampling=True))
    inputs = _make_inputs(batch_size=4)
    data_samples = _make_data_samples(4, cls_name='grid')

    class _FakeAnomalyGenerator:
        def __init__(self):
            self.calls = 0

        def generate_anomaly(self, img):
            self.calls += 1
            mask = np.zeros(img.shape[:2], dtype=np.float32)
            return img.astype(np.uint8), mask

    fake_generator = _FakeAnomalyGenerator()
    monkeypatch.setattr(model, '_get_anomaly_generator', lambda category=None: fake_generator)

    losses = model(inputs, data_samples, mode='loss')

    assert fake_generator.calls == 2
    assert set(losses.keys()) == {'loss', 'l1_loss', 'focal_loss'}
    assert all(torch.isfinite(value).all().item() for value in losses.values())


def test_forward_loss_generates_anomalies_before_center_crop(monkeypatch):
    model = MODELS.build(_make_cfg(
        alternate_anomaly_sampling=True,
        anomaly_source_resize=96,
        anomaly_source_crop=64,
    ))
    inputs = _make_inputs(batch_size=2)
    data_samples = _make_data_samples(2, cls_name='grid', height=64, width=64)

    loaded_shapes = []
    generated_shapes = []

    def _fake_load_training_source_image(img_path):
        del img_path
        loaded_shapes.append((96, 96, 3))
        return np.full((96, 96, 3), 127, dtype=np.uint8)

    class _FakeAnomalyGenerator:
        def generate_anomaly(self, img):
            generated_shapes.append(tuple(img.shape))
            mask = np.ones(img.shape[:2], dtype=np.float32)
            return img.copy(), mask

    monkeypatch.setattr('baoiad.models.detectors.memseg.os.path.isfile', lambda path: True)
    monkeypatch.setattr(model, '_load_training_source_image', _fake_load_training_source_image)
    monkeypatch.setattr(model, '_get_anomaly_generator', lambda category=None: _FakeAnomalyGenerator())

    losses = model(inputs, data_samples, mode='loss')

    assert loaded_shapes == [(96, 96, 3), (96, 96, 3)]
    assert generated_shapes == [(96, 96, 3)]
    assert set(losses.keys()) == {'loss', 'l1_loss', 'focal_loss'}
    assert all(torch.isfinite(value).all().item() for value in losses.values())


def test_predict_requires_initialized_memory_bank():
    model = MODELS.build(_make_cfg())
    inputs = _make_inputs(batch_size=2)
    data_samples = _make_data_samples(2)

    with pytest.raises(RuntimeError, match='memory bank'):
        model(inputs, data_samples, mode='predict')


def test_build_memory_bank_enables_finite_predict_outputs():
    model = MODELS.build(_make_cfg(nb_memory_sample=3))
    dataset = _IndexedDataset(num_items=5)
    loader = _LoaderWithDataset(dataset)
    model.build_memory_bank(loader)

    inputs = _make_inputs(batch_size=2)
    data_samples = _make_data_samples(2)
    outputs = model(inputs, data_samples, mode='predict')

    assert isinstance(outputs, list)
    assert len(outputs) == 2
    scores = torch.tensor([output.pred_score for output in outputs], dtype=torch.float32)
    maps = torch.stack([output.pred_anomaly_map for output in outputs])
    assert torch.isfinite(scores).all()
    assert torch.isfinite(maps).all()
    assert maps.abs().sum().item() > 0

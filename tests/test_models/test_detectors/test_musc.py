"""Tests for MuScDetector and MuSc CLIP backbone alignment guards."""

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

import baoiad  # noqa: F401
from baoiad.models.backbones import musc_clip_backbone as musc_clip_backbone_module
from baoiad.models.backbones.musc_clip_backbone import MuScCLIPBackbone
from baoiad.models.detectors.musc import MMO, MSM
from baoiad.registry import MODELS
from baoiad.structures import ADDataSample


def _make_data_samples(batch_size, height=32, width=32):
    samples = []
    for index in range(batch_size):
        sample = ADDataSample()
        sample.gt_label = index % 2
        sample.gt_mask = torch.zeros(height, width)
        sample.cls_name = 'bottle'
        sample.img_path = f'/fake/{index}.png'
        sample.defect_type = 'good'
        samples.append(sample)
    return samples


@MODELS.register_module(force=True)
class ToyMuScBackbone(BaseModule):
    """Small deterministic backbone for MuSc detector tests."""

    def __init__(self, feature_layers=None, width: int = 4):
        super().__init__()
        self.feature_layers = feature_layers or [1, 2]
        self.resolved_feature_layers = list(self.feature_layers)
        self.width = width
        self.anchor = nn.Parameter(torch.zeros(1), requires_grad=False)

    def encode_image(self, x, out_layers=None):
        out_layers = out_layers or self.feature_layers
        batch_size = x.shape[0]
        device = x.device

        image_features = torch.stack([
            x.mean(dim=(1, 2, 3)),
            x.amax(dim=(1, 2, 3)),
            x.amin(dim=(1, 2, 3)),
            torch.arange(batch_size, device=device, dtype=torch.float32) + 1.0,
        ], dim=1)
        image_features = F.normalize(image_features, dim=-1)

        patch_tokens = []
        batch_offset = torch.arange(batch_size, device=device, dtype=torch.float32).view(batch_size, 1)
        spatial_template = torch.arange(1, 5, device=device, dtype=torch.float32).view(1, 4)
        for layer_index, _ in enumerate(out_layers):
            tokens = torch.zeros(batch_size, 5, self.width, device=device)
            tokens[:, 0, 0] = 1.0
            tokens[:, 1:, layer_index % self.width] = spatial_template + batch_offset
            tokens[:, 1:, (layer_index + 1) % self.width] = (spatial_template * 0.25) + batch_offset
            patch_tokens.append(tokens.cpu())

        return image_features, patch_tokens


class _FakeRefOpenClipModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.last_out_layers = None

    def encode_image(self, x, out_layers):
        self.last_out_layers = list(out_layers)
        batch_size = x.shape[0]
        image_features = torch.ones(batch_size, 768, device=x.device)
        patch_tokens = [
            torch.full((batch_size, 5, 1024), float(layer), device=x.device)
            for layer in out_layers
        ]
        return image_features, patch_tokens


class _FakeRefOpenClipModule:
    def __init__(self):
        self.model = None

    def create_model_and_transforms(self, model_name, image_size, pretrained):
        self.model = _FakeRefOpenClipModel()
        return self.model, None, None


def test_musc_forward_tensor_and_loss_with_toy_backbone():
    model = MODELS.build(dict(
        type='MuScDetector',
        backbone=dict(type='ToyMuScBackbone', feature_layers=[1, 2], width=4),
        feature_layers=[1, 2],
        r_list=[1],
        image_size=32,
        topmin_min=0.0,
        topmin_max=0.5,
        k_list=[1],
    ))

    tensor_out = model(torch.randn(2, 3, 32, 32), mode='tensor')
    assert len(tensor_out) == 2
    assert tensor_out[0].shape == (2, 4)
    assert len(tensor_out[1]) == 2

    loss_out = model(torch.randn(2, 3, 32, 32), _make_data_samples(2), mode='loss')
    assert 'loss' in loss_out
    assert torch.isfinite(loss_out['loss'])


def test_musc_score_all_updates_placeholder_predictions():
    model = MODELS.build(dict(
        type='MuScDetector',
        backbone=dict(type='ToyMuScBackbone', feature_layers=[1, 2], width=4),
        feature_layers=[1, 2],
        r_list=[1],
        image_size=8,
        topmin_min=0.0,
        topmin_max=0.5,
        k_list=[1],
    ))
    model.eval()

    data_samples = _make_data_samples(2, height=8, width=8)
    placeholders = model(torch.randn(2, 3, 8, 8), data_samples, mode='predict')
    assert len(placeholders) == 2
    assert placeholders[0].pred_score == 0.0
    assert torch.count_nonzero(placeholders[0].pred_anomaly_map) == 0

    finalized = model.score_all()
    assert len(finalized) == 2
    for sample in finalized:
        assert math.isfinite(float(sample.pred_score))
        assert tuple(sample.pred_anomaly_map.shape) == (1, 8, 8)
        assert torch.isfinite(sample.pred_anomaly_map).all()
    assert torch.count_nonzero(finalized[0].pred_anomaly_map) > 0


def test_msm_handles_single_image_without_crashing():
    scores = MSM(torch.randn(1, 4, 3), torch.device('cpu'), topmin_min=0.0, topmin_max=0.3)
    assert scores.shape == (1, 4)
    assert torch.count_nonzero(scores) == 0


def test_mmo_skips_invalid_k_values_for_small_batches():
    similarity = torch.eye(2)
    score = torch.tensor([0.2, 0.8], dtype=torch.float32)
    refined = MMO(similarity, score, [1, 2, 3])
    assert refined.shape == score.shape
    assert torch.isfinite(refined).all()


def test_musc_score_all_handles_default_small_batch_hyperparameters():
    model = MODELS.build(dict(
        type='MuScDetector',
        backbone=dict(type='ToyMuScBackbone', feature_layers=[1, 2], width=4),
        feature_layers=[1, 2],
        r_list=[1],
        image_size=8,
        topmin_min=0.0,
        topmin_max=0.3,
        k_list=[1, 2, 3],
    ))
    model.eval()

    data_samples = _make_data_samples(1, height=8, width=8)
    placeholders = model(torch.randn(1, 3, 8, 8), data_samples, mode='predict')
    assert len(placeholders) == 1

    finalized = model.score_all()
    assert len(finalized) == 1
    assert math.isfinite(float(finalized[0].pred_score))
    assert tuple(finalized[0].pred_anomaly_map.shape) == (1, 8, 8)
    assert torch.isfinite(finalized[0].pred_anomaly_map).all()


def test_musc_clip_backbone_uses_reference_layer_offset(monkeypatch):
    fake_open_clip = _FakeRefOpenClipModule()
    monkeypatch.setattr(
        musc_clip_backbone_module,
        '_try_import_musc_open_clip',
        lambda: (fake_open_clip, True, None),
    )

    backbone = MuScCLIPBackbone(
        model_name='ViT-L-14-336',
        pretrained='openai',
        feature_layers=[5, 11, 17, 23],
        image_size=56,
        frozen=True,
        use_ref_open_clip=True,
        require_ref_open_clip=True,
    )
    backbone.encode_image(torch.randn(1, 3, 56, 56))

    assert backbone.resolved_feature_layers == [6, 12, 18, 24]
    assert fake_open_clip.model is not None
    assert fake_open_clip.model.last_out_layers == [6, 12, 18, 24]


def test_musc_clip_backbone_can_require_reference_open_clip(monkeypatch):
    monkeypatch.setattr(
        musc_clip_backbone_module,
        '_try_import_musc_open_clip',
        lambda: (None, False, 'reference import failed'),
    )

    with pytest.raises(RuntimeError, match='reference import failed'):
        MuScCLIPBackbone(
            model_name='ViT-L-14-336',
            pretrained='openai',
            feature_layers=[5, 11, 17, 23],
            image_size=56,
            frozen=True,
            use_ref_open_clip=True,
            require_ref_open_clip=True,
        )

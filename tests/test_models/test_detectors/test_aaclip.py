"""Tests for AACLIPDetector."""

from __future__ import annotations

from pathlib import Path
from unittest import TestCase

import pytest
import torch

from baoiad.structures import ADDataSample

import baoiad  # noqa: F401
from baoiad.models.detectors import aaclip as aaclip_module
from baoiad.registry import MODELS


def _make_data_samples(batch_size, class_names=None, h=8, w=8):
    class_names = class_names or ['bottle'] * batch_size
    samples = []
    for i in range(batch_size):
        sample = ADDataSample()
        sample.gt_label = i % 2
        sample.gt_mask = torch.zeros(h, w)
        sample.set_metainfo({
            'cls_name': class_names[i],
            'img_path': f'/fake/{i}.png',
            'defect_type': 'good',
        })
        samples.append(sample)
    return samples


class _FakeResBlock(torch.nn.Module):
    def forward(self, x, attn_mask=None):
        del attn_mask
        return x, None


class _FakeTransformer(torch.nn.Module):
    def __init__(self, depth):
        super().__init__()
        self.resblocks = torch.nn.ModuleList([_FakeResBlock() for _ in range(depth)])

    def get_cast_dtype(self):
        return torch.float32


class _FakeVisual(torch.nn.Module):
    def __init__(self, image_size):
        super().__init__()
        self.image_size = image_size
        self.conv1 = torch.nn.Conv2d(3, 1024, kernel_size=1, stride=1, bias=False)
        num_patches = image_size * image_size
        self.class_embedding = torch.nn.Parameter(torch.zeros(1024))
        self.positional_embedding = torch.nn.Parameter(torch.zeros(num_patches + 1, 1024))
        self.patch_dropout = torch.nn.Identity()
        self.ln_pre = torch.nn.Identity()
        self.ln_post = torch.nn.Identity()
        self.proj = torch.nn.Parameter(torch.randn(1024, 768) * 0.01)
        self.transformer = _FakeTransformer(24)

    def DAPM_replace(self, DPAM_layer):
        del DPAM_layer

    def _global_pool(self, tokens):
        return tokens[:, 0], tokens[:, 1:]


class _FakeClipModel(torch.nn.Module):
    def __init__(self, image_size):
        super().__init__()
        self.visual = _FakeVisual(image_size)
        self.token_embedding = torch.nn.Embedding(512, 768)
        self.positional_embedding = torch.nn.Parameter(torch.zeros(77, 768))
        self.transformer = _FakeTransformer(12)
        self.attn_mask = None
        self.ln_final = torch.nn.Identity()

    def encode_image(self, image, out_layers):
        x = self.visual.conv1(image)
        patches = x.flatten(2).permute(0, 2, 1)
        cls = self.visual.class_embedding.to(image.dtype).view(1, 1, -1).expand(image.shape[0], -1, -1)
        tokens = torch.cat([cls, patches], dim=1)
        tokens = tokens + self.visual.positional_embedding.unsqueeze(0).to(tokens.dtype)
        cls_token = tokens[:, 1:, :].mean(dim=1) @ self.visual.proj
        return cls_token, [tokens.clone() for _ in out_layers]

    def encode_text(self, text):
        embedded = self.token_embedding(text)
        return embedded.mean(dim=1)


def _fake_tokenize(sentences):
    tokens = torch.zeros(len(sentences), 77, dtype=torch.long)
    for index, sentence in enumerate(sentences):
        tokens[index, -1] = max(1, min(255, len(sentence)))
    return tokens


def _fake_reference_api():
    def _create_model(model_name, img_size, pretrained, device, require_pretrained):
        del model_name, pretrained, device, require_pretrained
        return _FakeClipModel(img_size)

    return dict(
        class_names={
            'MVTec': ['bottle', 'cable'],
            'VisA': ['candle', 'pcb1'],
        },
        domains={
            'MVTec': 'Industrial',
            'VisA': 'Industrial',
        },
        prompts={
            'prompt_normal': ['{}', 'a {}'],
            'prompt_abnormal': ['damaged {}', '{} with defect'],
            'prompt_templates': ['{}.', 'a photo of {}.'],
        },
        real_names={
            'MVTec': {
                'bottle': 'bottle',
                'cable': 'cable',
            },
            'VisA': {
                'candle': 'candle',
                'pcb1': 'pcb1',
            },
        },
        tokenize=_fake_tokenize,
        create_model=_create_model,
    )


@pytest.fixture(autouse=True)
def _patch_reference_api(monkeypatch):
    monkeypatch.setattr(
        aaclip_module,
        '_import_reference_api',
        lambda reference_root: _fake_reference_api(),
    )


class TestAACLIPDetector(TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.base_cfg = dict(
            type='AACLIPDetector',
            clip_model='ViT-L-14-336',
            pretrained='openai',
            image_size=8,
            reference_root='.refs/AA-CLIP',
            levels=[6],
            text_adapt_until=1,
            image_adapt_until=1,
            require_official_assets=False,
        )

    def test_text_stage_loss_contract(self):
        model = MODELS.build({
            **self.base_cfg,
            'training_stage': 'text',
        })
        out = model(
            torch.randn(2, 3, 8, 8),
            _make_data_samples(2, ['candle', 'pcb1']),
            mode='loss',
        )
        assert 'loss' in out
        assert torch.isfinite(out['loss'])

    def test_image_stage_loads_text_adapter_from_mmengine_checkpoint(self):
        stage1_model = MODELS.build({
            **self.base_cfg,
            'training_stage': 'text',
        })
        ckpt_path = Path(self._testMethodName + '_text.pth')
        checkpoint = {
            'state_dict': {
                f'adapted_model.text_adapter.{key}': value.detach().clone()
                for key, value in stage1_model.adapted_model.text_adapter.state_dict().items()
            }
        }
        torch.save(checkpoint, ckpt_path)
        try:
            model = MODELS.build({
                **self.base_cfg,
                'training_stage': 'image',
                'text_adapter_ckpt': str(ckpt_path),
            })
            out = model(
                torch.randn(2, 3, 8, 8),
                _make_data_samples(2, ['candle', 'pcb1']),
                mode='loss',
            )
            assert 'loss' in out
            assert torch.isfinite(out['loss'])
        finally:
            ckpt_path.unlink(missing_ok=True)

    def test_image_adapter_legacy_key_remap(self):
        source_model = MODELS.build({
            **self.base_cfg,
            'training_stage': 'image',
        })
        legacy_state = {}
        for key, value in source_model.adapted_model.image_adapter.state_dict().items():
            legacy_key = key.replace('.fc.0.weight', '.fc.weight')
            legacy_state[legacy_key] = value.detach().clone()
        ckpt_path = Path(self._testMethodName + '_image.pth')
        torch.save({'image_adapter': legacy_state}, ckpt_path)
        try:
            model = MODELS.build({
                **self.base_cfg,
                'training_stage': 'inference',
                'image_adapter_ckpt': str(ckpt_path),
            })
            out = model(
                torch.randn(2, 3, 8, 8),
                _make_data_samples(2, ['bottle', 'cable']),
                mode='predict',
            )
            assert len(out) == 2
            assert all(torch.isfinite(sample.pred_anomaly_map).all() for sample in out)
        finally:
            ckpt_path.unlink(missing_ok=True)

    def test_predict_multi_class(self):
        model = MODELS.build({
            **self.base_cfg,
            'training_stage': 'none',
        })
        out = model(
            torch.randn(2, 3, 8, 8),
            _make_data_samples(2, ['bottle', 'cable']),
            mode='predict',
        )
        assert len(out) == 2
        assert all(isinstance(sample.pred_score, float) for sample in out)
        assert all(torch.isfinite(sample.pred_anomaly_map).all() for sample in out)

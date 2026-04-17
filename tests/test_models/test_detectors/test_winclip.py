"""Tests for WinClipDetector."""

from types import SimpleNamespace
from unittest import TestCase

import torch
import torch.nn as nn

from baoiad.structures import ADDataSample

import baoiad  # noqa: F401
from baoiad.registry import MODELS


class _DummyVisual(nn.Module):
    def __init__(self, embed_dim=4):
        super().__init__()
        self.output_tokens = True
        self.grid_size = (15, 15)
        self.conv1 = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16, bias=False)
        self.patch_dropout = nn.Identity()
        self.ln_pre = nn.Identity()
        self.transformer = nn.Identity()
        self.ln_post = nn.Identity()
        self.proj = None
        self.positional_embedding = nn.Parameter(
            torch.zeros(1 + self.grid_size[0] * self.grid_size[1], embed_dim)
        )

    def _global_pool(self, tokens):
        return tokens[:, 0], tokens[:, 1:]


class _DummyClipModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = _DummyVisual()
        self.last_image_input = None

    def encode_image(self, batch):
        self.last_image_input = batch.detach().clone()
        num_patches = self.visual.grid_size[0] * self.visual.grid_size[1]
        scalar = batch.mean(dim=(1, 2, 3))
        cls_token = torch.stack(
            [scalar, scalar + 0.1, scalar + 0.2, scalar + 0.3], dim=1
        )
        patch_offsets = torch.linspace(
            0.0, 0.2, num_patches, device=batch.device, dtype=batch.dtype
        ).view(1, num_patches, 1)
        patch_tokens = cls_token.unsqueeze(1) + patch_offsets
        tokens = torch.cat([cls_token.unsqueeze(1), patch_tokens], dim=1)
        tokens = self.visual.patch_dropout(tokens)
        return tokens[:, 0], tokens[:, 1:]

    def encode_text(self, tokens):
        tokens = tokens.float()
        token_sum = tokens.sum(dim=1)
        token_delta = tokens[:, 0] - tokens[:, 1]
        return torch.stack(
            [tokens[:, 0], tokens[:, 1], token_sum, token_delta], dim=1
        )


if 'DummyWinClipBackbone' not in MODELS.module_dict:
    @MODELS.register_module()
    class DummyWinClipBackbone(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.model = _DummyClipModel()
            self.preprocess = SimpleNamespace(transforms=[])

        def tokenize(self, prompts):
            rows = []
            for prompt in prompts:
                prompt = str(prompt)
                rows.append([
                    float(len(prompt)),
                    float(sum(ord(char) for char in prompt) % 997),
                ])
            return torch.tensor(rows, dtype=torch.float32)


def _make_data_samples(class_names, height=32, width=32):
    samples = []
    for idx, class_name in enumerate(class_names):
        sample = ADDataSample()
        sample.gt_label = idx % 2
        sample.gt_mask = torch.zeros(height, width)
        sample.cls_name = class_name
        sample.img_path = f'/fake/{class_name}_{idx}.png'
        sample.defect_type = 'good'
        samples.append(sample)
    return samples


class TestWinClipDetector(TestCase):
    def _build_model(self, **overrides):
        cfg = dict(
            type='WinClipDetector',
            class_name='object',
            scales=(2,),
            k_shot=0,
            apply_transform=False,
            backbone=dict(type='DummyWinClipBackbone'),
        )
        cfg.update(overrides)
        model = MODELS.build(cfg)
        model.eval()
        return model

    def test_pe_adaptation_matches_256_input_path(self):
        model = self._build_model()
        self.assertEqual(model.grid_size, (16, 16))
        self.assertEqual(model.clip.visual.grid_size, (16, 16))
        self.assertEqual(model.clip.visual.positional_embedding.shape[0], 257)
        self.assertEqual(tuple(model._masks[0].shape), (4, 225))

    def test_apply_transform_false_renormalizes_imagenet_inputs_for_clip(self):
        model = self._build_model()
        raw = torch.tensor(
            [[[0.2, 0.4], [0.6, 0.8]],
             [[0.1, 0.3], [0.5, 0.7]],
             [[0.9, 0.2], [0.4, 0.6]]],
            dtype=torch.float32,
        ).unsqueeze(0)
        imagenet = (raw - model._imagenet_mean) / model._imagenet_std
        expected = (raw - model._clip_mean) / model._clip_std

        model._encode_image(imagenet)

        torch.testing.assert_close(model.clip.last_image_input, expected)

    def test_forward_uses_data_sample_class_names_instead_of_default_prompt(self):
        model = self._build_model()
        calls = []
        original = model._get_text_embeddings_for_class

        def wrapped(class_name, device):
            calls.append(class_name)
            return original(class_name, device)

        model._get_text_embeddings_for_class = wrapped
        inputs = torch.randn(2, 3, 32, 32)
        samples = _make_data_samples(['bottle', 'cable'])

        model(inputs, samples, mode='predict')

        self.assertEqual(calls, ['bottle', 'cable'])
        self.assertIn('bottle', model._text_embedding_cache)
        self.assertIn('cable', model._text_embedding_cache)
        self.assertNotIn('object', calls)

    def test_forward_predict_returns_finite_scores_and_maps(self):
        model = self._build_model()
        inputs = torch.randn(2, 3, 32, 32)
        samples = _make_data_samples(['bottle', 'capsule'])

        results = model(inputs, samples, mode='predict')

        self.assertEqual(len(results), 2)
        for result in results:
            self.assertTrue(torch.isfinite(torch.tensor(result.pred_score)))
            self.assertEqual(tuple(result.pred_anomaly_map.shape), (1, 32, 32))
            self.assertTrue(torch.isfinite(result.pred_anomaly_map).all())

    def test_forward_loss_returns_zero_like_training_stub(self):
        model = self._build_model()
        inputs = torch.randn(2, 3, 32, 32)
        samples = _make_data_samples(['bottle', 'cable'])

        losses = model(inputs, samples, mode='loss')

        self.assertIn('loss', losses)
        self.assertTrue(torch.isfinite(losses['loss']))

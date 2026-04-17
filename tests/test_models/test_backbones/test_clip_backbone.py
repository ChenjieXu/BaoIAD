"""Tests for OpenCLIPBackbone."""

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import torch.nn as nn

from baoiad.models.backbones.clip_backbone import OpenCLIPBackbone


class _FakeClipModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = SimpleNamespace(grid_size=(15, 15))


class TestOpenCLIPBackbone(TestCase):
    def test_open_clip_kwargs_are_forwarded(self):
        calls = {}

        def fake_create_model_and_transforms(model_name, **kwargs):
            calls['model_name'] = model_name
            calls['kwargs'] = kwargs
            return _FakeClipModel(), None, 'preprocess'

        fake_open_clip = SimpleNamespace(
            __name__='open_clip',
            create_model_and_transforms=fake_create_model_and_transforms,
            get_tokenizer=lambda model_name: f'tokenizer:{model_name}',
            tokenize=lambda texts: texts,
        )

        with patch(
            'baoiad.models.backbones.clip_backbone._import_open_clip',
            return_value=fake_open_clip,
        ):
            backbone = OpenCLIPBackbone(
                model_name='ViT-B-16-plus-240',
                pretrained='/tmp/openclip.pt',
                cache_dir='/tmp/openclip-cache',
                pretrained_image_path='/tmp/image.pt',
                pretrained_text_path='/tmp/text.pt',
                image_size=240,
                load_weights=False,
                frozen=True,
            )

        self.assertEqual(calls['model_name'], 'ViT-B-16-plus-240')
        self.assertEqual(calls['kwargs']['pretrained'], '/tmp/openclip.pt')
        self.assertEqual(calls['kwargs']['cache_dir'], '/tmp/openclip-cache')
        self.assertEqual(calls['kwargs']['pretrained_image_path'], '/tmp/image.pt')
        self.assertEqual(calls['kwargs']['pretrained_text_path'], '/tmp/text.pt')
        self.assertEqual(calls['kwargs']['force_image_size'], 240)
        self.assertFalse(calls['kwargs']['load_weights'])
        self.assertEqual(backbone.tokenize(['hello']), ['hello'])

"""Tests for TIMMBackbone."""

import os
import sys
import tempfile
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import torch
import torch.nn as nn

from baoiad.models.backbones.timm_backbone import TIMMBackbone


class _FakeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_info = SimpleNamespace(
            channels=lambda: (8,),
            reduction=lambda: (16,),
        )
        self.num_features = 16
        self.loaded_state_dict = None
        self.conv1 = nn.Conv2d(3, 8, kernel_size=1)
        self.layer1 = nn.Sequential(nn.Conv2d(8, 8, kernel_size=1))
        self.layer2 = nn.Sequential(nn.Conv2d(8, 8, kernel_size=1))
        self.layer3 = nn.Sequential(nn.Conv2d(8, 8, kernel_size=1))
        self.layer4 = nn.Sequential(nn.Conv2d(8, 8, kernel_size=1))

    def forward(self, x):
        return [x]

    def forward_features(self, x):
        return x + 1

    def forward_head(self, x, pre_logits=False):
        if pre_logits:
            return x.mean(dim=(2, 3))
        return x

    def load_state_dict(self, state_dict, strict=True):
        self.loaded_state_dict = (state_dict, strict)


class _FakePretrainedCfg:
    def __init__(self, url, hf_hub_id='', hf_hub_filename=None):
        self.url = url
        self.hf_hub_id = hf_hub_id
        self.hf_hub_filename = hf_hub_filename


class TestTIMMBackbone(TestCase):
    def test_cached_checkpoint_is_used_before_remote_pretrained(self):
        calls = {}
        fake_net = _FakeNet()

        def fake_create_model(model_name, pretrained, features_only, out_indices):
            calls['model_name'] = model_name
            calls['pretrained'] = pretrained
            calls['features_only'] = features_only
            calls['out_indices'] = out_indices
            return fake_net

        fake_timm = SimpleNamespace(
            create_model=fake_create_model,
            get_pretrained_cfg=lambda _: _FakePretrainedCfg(
                'https://example.com/tf_efficientnet_b4_aa-818f208c.pth',
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = os.path.join(tmpdir, 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(
                checkpoint_dir,
                'tf_efficientnet_b4_aa-818f208c.pth',
            )
            torch.save(
                {'state_dict': {'module.conv.weight': torch.ones(1)}},
                checkpoint_path,
            )

            with patch.dict(sys.modules, {'timm': fake_timm}):
                with patch('torch.hub.get_dir', return_value=tmpdir):
                    backbone = TIMMBackbone(
                        model_name='tf_efficientnet_b4.aa_in1k',
                        pretrained=True,
                        features_only=True,
                        out_indices=(4,),
                        frozen=False,
                    )

        self.assertEqual(calls['model_name'], 'tf_efficientnet_b4.aa_in1k')
        self.assertFalse(calls['pretrained'])
        self.assertTrue(calls['features_only'])
        self.assertEqual(calls['out_indices'], [4])
        self.assertEqual(backbone.out_channels, (8,))
        state_dict, strict = fake_net.loaded_state_dict
        self.assertFalse(strict)
        self.assertIn('conv.weight', state_dict)

    def test_load_checkpoint_uses_baoiad_policy(self):
        fake_net = _FakeNet()
        backbone = TIMMBackbone.__new__(TIMMBackbone)
        nn.Module.__init__(backbone)
        backbone.net = fake_net

        with patch(
            'baoiad.models.backbones.timm_backbone.load_baoiad_checkpoint',
            return_value={'state_dict': {'module.weight': torch.ones(1)}},
        ) as mocked_load:
            TIMMBackbone._load_checkpoint(backbone, 'dummy.pth', strict=False)

        mocked_load.assert_called_once_with('dummy.pth', map_location='cpu')
        state_dict, strict = fake_net.loaded_state_dict
        self.assertFalse(strict)
        self.assertIn('weight', state_dict)

    def test_hf_cache_path_is_used_when_torch_hub_checkpoint_is_missing(self):
        fake_timm = SimpleNamespace(
            get_pretrained_cfg=lambda _: _FakePretrainedCfg(
                'https://example.com/wide_resnet50_racm-8234f177.pth',
                hf_hub_id='timm/wide_resnet50_2.racm_in1k',
            ),
        )

        with patch('torch.hub.get_dir', return_value='/tmp/torch-hub'):
            with patch('huggingface_hub.try_to_load_from_cache', return_value='/tmp/model.safetensors'):
                with patch('os.path.exists', side_effect=lambda path: path == '/tmp/model.safetensors'):
                    path = TIMMBackbone._resolve_cached_pretrained_path(fake_timm, 'wide_resnet50_2.racm_in1k')

        self.assertEqual(path, '/tmp/model.safetensors')

    def test_wrn50_default_prefers_matching_pretrained_cfg_when_legacy_fallback_disabled(self):
        fake_timm = SimpleNamespace(
            get_pretrained_cfg=lambda _: _FakePretrainedCfg(
                'https://example.com/wide_resnet50_racm-8234f177.pth',
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = os.path.join(tmpdir, 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            legacy_path = os.path.join(checkpoint_dir, 'wide_resnet50_2-95faca4d.pth')
            racm_path = os.path.join(checkpoint_dir, 'wide_resnet50_racm-8234f177.pth')
            torch.save({'state_dict': {'weight': torch.ones(1)}}, legacy_path)
            torch.save({'state_dict': {'weight': torch.ones(1)}}, racm_path)

            with patch('torch.hub.get_dir', return_value=tmpdir):
                legacy_default = TIMMBackbone._resolve_cached_pretrained_path(
                    fake_timm,
                    'wide_resnet50_2',
                    allow_legacy_fallback=True,
                )
                strict_default = TIMMBackbone._resolve_cached_pretrained_path(
                    fake_timm,
                    'wide_resnet50_2',
                    allow_legacy_fallback=False,
                )

        self.assertEqual(legacy_default, legacy_path)
        self.assertEqual(strict_default, racm_path)

    def test_load_checkpoint_supports_safetensors(self):
        fake_net = _FakeNet()
        backbone = TIMMBackbone.__new__(TIMMBackbone)
        nn.Module.__init__(backbone)
        backbone.net = fake_net

        with patch(
            'baoiad.models.backbones.timm_backbone.load_baoiad_checkpoint',
            return_value={'module.weight': torch.ones(1)},
        ) as mocked_load:
            TIMMBackbone._load_checkpoint(backbone, 'dummy.safetensors', strict=False)

        mocked_load.assert_called_once_with(
            'dummy.safetensors', map_location='cpu')
        state_dict, strict = fake_net.loaded_state_dict
        self.assertFalse(strict)
        self.assertIn('weight', state_dict)

    def test_full_model_mode_exposes_pre_logits(self):
        calls = {}
        fake_net = _FakeNet()

        def fake_create_model(model_name, pretrained, features_only):
            calls['model_name'] = model_name
            calls['pretrained'] = pretrained
            calls['features_only'] = features_only
            return fake_net

        fake_timm = SimpleNamespace(create_model=fake_create_model)

        with patch.dict(sys.modules, {'timm': fake_timm}):
            backbone = TIMMBackbone(
                model_name='tf_efficientnet_b4.aa_in1k',
                pretrained=False,
                features_only=False,
                frozen=False,
            )

        self.assertEqual(calls['model_name'], 'tf_efficientnet_b4.aa_in1k')
        self.assertFalse(calls['pretrained'])
        self.assertFalse(calls['features_only'])
        self.assertEqual(backbone.out_channels, (8,))
        self.assertEqual(backbone.num_features, 16)
        inputs = torch.randn(2, 16, 4, 4)
        pre_logits = backbone.forward_pre_logits(inputs)
        self.assertEqual(tuple(pre_logits.shape), (2, 16))

    def test_partial_freeze_keeps_named_stages_frozen(self):
        fake_net = _FakeNet()
        fake_timm = SimpleNamespace(create_model=lambda **_: fake_net)

        with patch.dict(sys.modules, {'timm': fake_timm}):
            backbone = TIMMBackbone(
                model_name='resnet18',
                pretrained=False,
                features_only=True,
                out_indices=(0,),
                frozen=False,
                frozen_names=('layer1', 'layer2', 'layer3'),
            )

        assert any(param.requires_grad for param in backbone.net.conv1.parameters())
        assert not any(param.requires_grad for param in backbone.net.layer1.parameters())
        assert not any(param.requires_grad for param in backbone.net.layer2.parameters())
        assert not any(param.requires_grad for param in backbone.net.layer3.parameters())
        assert any(param.requires_grad for param in backbone.net.layer4.parameters())

        backbone.train(True)
        assert backbone.net.layer1.training is False
        assert backbone.net.layer2.training is False
        assert backbone.net.layer3.training is False
        assert backbone.net.layer4.training is True

    def test_partial_freeze_can_keep_frozen_stages_in_train_mode(self):
        fake_net = _FakeNet()
        fake_timm = SimpleNamespace(create_model=lambda **_: fake_net)

        with patch.dict(sys.modules, {'timm': fake_timm}):
            backbone = TIMMBackbone(
                model_name='resnet18',
                pretrained=False,
                features_only=True,
                out_indices=(0,),
                frozen=False,
                frozen_names=('layer1', 'layer2', 'layer3'),
                frozen_names_eval=False,
            )

        backbone.train(True)
        assert backbone.net.layer1.training is True
        assert backbone.net.layer2.training is True
        assert backbone.net.layer3.training is True
        assert backbone.net.layer4.training is True
        assert not any(param.requires_grad for param in backbone.net.layer1.parameters())

"""Tests for RegADDetector."""

import types
from pathlib import Path

import torch
from mmengine import Config

import baoiad  # noqa: F401

from baoiad.registry import MODELS
from baoiad.structures import ADDataSample

ROOT = Path(__file__).resolve().parents[3]


def _make_data_samples(batch_size, H=64, W=64, shot=2):
    samples = []
    for i in range(batch_size):
        sample = ADDataSample()
        sample.gt_label = i % 2
        sample.gt_mask = torch.zeros(H, W)
        sample.set_metainfo({
            'cls_name': 'bottle',
            'img_path': f'/fake/{i}.png',
            'defect_type': 'good',
        })
        sample.support_imgs = torch.rand(shot, 3, H, W)
        samples.append(sample)
    return samples


def _lightweight_extract_features(_self, x):
    batch_size = x.shape[0]
    device = x.device
    return {
        'layer1': torch.ones(batch_size, 4, 2, 2, device=device),
        'layer2': torch.ones(batch_size, 6, 2, 2, device=device),
        'layer3': torch.ones(batch_size, 8, 2, 2, device=device),
        'final_feat': torch.ones(batch_size, 256, 2, 2, device=device),
    }


def _lightweight_build_memory_bank(self, _dataloader=None):
    device = next(self.parameters()).device
    self.support_feat = torch.zeros(1, 256, 2, 2, device=device)
    self.embedding_mean = torch.zeros(8, 4, device=device)
    self.embedding_cov_inv = torch.eye(8, device=device).unsqueeze(-1).repeat(1, 1, 4)


def _patch_lightweight_regad(model):
    model.extract_features = types.MethodType(_lightweight_extract_features, model)
    model.build_memory_bank = types.MethodType(_lightweight_build_memory_bank, model)
    model.fit = types.MethodType(lambda self, *args, **kwargs: _lightweight_build_memory_bank(self), model)
    return model


def test_regad_forward_tensor():
    model = MODELS.build(
        dict(
            type='RegADDetector',
            backbone='resnet18',
            layers=(3,),
            few_shot=2,
            img_size=16,
            pretrained_backbone=False,
        )
    )
    _patch_lightweight_regad(model)
    model.eval()
    out = model(torch.randn(2, 3, 16, 16), mode='tensor')
    assert isinstance(out, dict)
    assert 'final_feat' in out


def test_regad_forward_loss_uses_support_imgs():
    model = MODELS.build(
        dict(
            type='RegADDetector',
            backbone='resnet18',
            layers=(3,),
            few_shot=2,
            img_size=16,
            pretrained_backbone=False,
        )
    )
    _patch_lightweight_regad(model)
    model.train()
    data_samples = _make_data_samples(2, 16, 16, shot=2)
    out = model(torch.randn(2, 3, 16, 16), data_samples, mode='loss')
    assert isinstance(out, dict)
    assert 'loss' in out
    assert torch.isfinite(out['loss']).item()


def test_regad_build_memory_bank_from_cached_supports_enables_predict():
    model = MODELS.build(
        dict(
            type='RegADDetector',
            backbone='resnet18',
            layers=(3,),
            few_shot=2,
            img_size=16,
            pretrained_backbone=False,
        )
    )
    _patch_lightweight_regad(model)
    model.train()
    for _ in range(2):
        model(torch.randn(2, 3, 16, 16), _make_data_samples(2, 16, 16, shot=2), mode='loss')
    model.build_memory_bank()
    model.eval()
    out = model(torch.randn(2, 3, 16, 16), _make_data_samples(2, 16, 16, shot=2), mode='predict')
    assert isinstance(out, list)
    assert len(out) == 2
    assert all(hasattr(sample, 'pred_score') for sample in out)


def test_regad_predict_requires_memory_bank():
    model = MODELS.build(
        dict(
            type='RegADDetector',
            backbone='resnet18',
            layers=(3,),
            few_shot=2,
            img_size=16,
            pretrained_backbone=False,
        )
    )
    _patch_lightweight_regad(model)
    model.eval()
    try:
        model(torch.randn(1, 3, 16, 16), _make_data_samples(1, 16, 16, shot=2), mode='predict')
    except RuntimeError as exc:
        assert 'build_support_bank_from_images()/fit()' in str(exc)
    else:
        raise AssertionError('predict() should fail before building the support bank.')


def test_regad_alignment_config_requires_official_support_set():
    cfg = Config.fromfile(ROOT / 'configs' / 'regad' / 'regad_wrn50_256_mvtec_strict.py')
    assert cfg.support_set_root.endswith('data/regad_official/support_set')
    assert cfg.strict_require_official_support_set is True
    assert cfg.benchmark_train_script == 'tools/train_regad_strict.py'
    assert cfg.benchmark_keep_dataloader_workers is True
    assert cfg.benchmark_result_selector == {
        'mode': 'best_balanced',
        'metrics': ['image_auroc', 'pixel_auroc'],
    }
    assert cfg.benchmark_resume_existing is True
    assert cfg.official_seed == 668
    assert cfg.shot == 4
    assert cfg.inferences == 10

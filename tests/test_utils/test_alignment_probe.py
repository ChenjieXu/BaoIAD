"""Tests for alignment probe helpers."""

import torch

import baoiad  # noqa: F401
from baoiad.models.predict_utils import build_predict_results
from baoiad.registry import DATASETS, MODELS
from baoiad.structures import ADDataSample
from baoiad.utils.alignment_probe import _prepare_dataloader_cfg, probe_config


@DATASETS.register_module(force=True)
class ProbeToyDataset:
    def __init__(self, length=4, image_size=16):
        self.length = length
        self.image_size = image_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample = ADDataSample()
        sample.set_metainfo({
            'cls_name': 'bottle',
            'img_path': f'/fake/{idx}.png',
            'defect_type': 'good',
        })
        sample.gt_label = 0
        sample.gt_mask = torch.zeros(self.image_size, self.image_size)
        value = float(idx + 1) / float(self.length + 1)
        return {
            'inputs': torch.full((3, self.image_size, self.image_size), value, dtype=torch.float32),
            'data_samples': sample,
        }


@MODELS.register_module(force=True)
class ProbeToyDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.memory_bank_ready = False
        self.normalization_ready = False
        self.pretrain_ready = False

    def pre_train_setup(self, dataloader):
        del dataloader
        self.pretrain_ready = True

    def build_memory_bank(self, dataloader=None):
        del dataloader
        self.memory_bank_ready = True

    def compute_normalization_stats(self, dataloader, device=None):
        del dataloader, device
        self.normalization_ready = True

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if isinstance(inputs, list):
            inputs = torch.stack(inputs)

        if mode == 'loss':
            return {'loss': inputs.mean() * 0 + self.weight.sum()}

        if mode == 'predict':
            if not self.memory_bank_ready:
                raise RuntimeError('memory bank is not built')
            scores = inputs.mean(dim=(1, 2, 3))
            return build_predict_results(data_samples, scores, inputs[:, :1])

        return inputs


@MODELS.register_module(force=True)
class ProbeGradWarmupDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.memory_bank_ready = False
        self.cached_loss_term = None

    def build_memory_bank(self, dataloader=None):
        del dataloader
        loss = self.cached_loss_term
        assert loss is not None
        loss.backward()
        self.memory_bank_ready = self.weight.grad is not None

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if isinstance(inputs, list):
            inputs = torch.stack(inputs)

        if mode == 'loss':
            self.cached_loss_term = inputs.mean() * self.weight
            return {'loss': self.cached_loss_term}

        if mode == 'predict':
            if not self.memory_bank_ready:
                raise RuntimeError('memory bank is not built')
            scores = inputs.mean(dim=(1, 2, 3))
            return build_predict_results(data_samples, scores, inputs[:, :1])

        return inputs


@MODELS.register_module(force=True)
class ProbeTemplateBuilderDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.template_ready = False

    def build_template_from_dataloader(self, dataloader, device):
        del device
        count = 0
        for _ in dataloader:
            count += 1
        self.template_ready = count > 0

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if isinstance(inputs, list):
            inputs = torch.stack(inputs)

        if mode == 'loss':
            return {'loss': inputs.mean() * 0 + self.weight.sum()}

        if mode == 'predict':
            if not self.template_ready:
                raise RuntimeError('template not built')
            scores = inputs.mean(dim=(1, 2, 3))
            return build_predict_results(data_samples, scores, inputs[:, :1])

        return inputs


def test_probe_config_runs_warmups_and_writes_json(tmp_path):
    config_path = tmp_path / 'probe_toy.py'
    config_path.write_text(
        '\n'.join([
            "default_scope = 'baoiad'",
            "randomness = dict(seed=42, deterministic=False)",
            "train_dataloader = dict(",
            "    batch_size=3,",
            "    num_workers=0,",
            "    sampler=dict(type='DefaultSampler', shuffle=False),",
            "    dataset=dict(type='ProbeToyDataset', length=4, image_size=16),",
            ")",
            "test_dataloader = dict(",
            "    batch_size=3,",
            "    num_workers=0,",
            "    sampler=dict(type='DefaultSampler', shuffle=False),",
            "    dataset=dict(type='ProbeToyDataset', length=4, image_size=16),",
            ")",
            "val_dataloader = test_dataloader",
            "model = dict(type='ProbeToyDetector')",
        ]),
        encoding='utf-8',
    )
    output_path = tmp_path / 'probe.json'

    report = probe_config(
        config_path=str(config_path),
        splits=('train', 'test'),
        max_batch_size=2,
        device='cpu',
        output=str(output_path),
    )

    assert report['passed'] is True
    assert report['splits']['train']['batch']['batch_size'] == 2
    assert report['splits']['train']['loss']['all_finite'] is True
    assert report['splits']['test']['memory_bank_warmup']['used'] is True
    assert report['splits']['test']['normalization_warmup']['used'] is True
    assert report['splits']['test']['predict']['score_stats']['finite'] is True
    assert output_path.is_file()


def test_probe_config_allows_grad_requiring_memory_bank_builders(tmp_path):
    config_path = tmp_path / 'probe_grad.py'
    config_path.write_text(
        '\n'.join([
            "default_scope = 'baoiad'",
            "randomness = dict(seed=42, deterministic=False)",
            "train_dataloader = dict(",
            "    batch_size=2,",
            "    num_workers=0,",
            "    sampler=dict(type='DefaultSampler', shuffle=False),",
            "    dataset=dict(type='ProbeToyDataset', length=4, image_size=16),",
            ")",
            "test_dataloader = dict(",
            "    batch_size=2,",
            "    num_workers=0,",
            "    sampler=dict(type='DefaultSampler', shuffle=False),",
            "    dataset=dict(type='ProbeToyDataset', length=4, image_size=16),",
            ")",
            "val_dataloader = test_dataloader",
            "model = dict(type='ProbeGradWarmupDetector')",
        ]),
        encoding='utf-8',
    )

    report = probe_config(
        config_path=str(config_path),
        splits=('train', 'test'),
        max_batch_size=2,
        device='cpu',
    )

    assert report['passed'] is True
    assert report['splits']['test']['memory_bank_warmup']['used'] is True
    assert report['splits']['test']['predict']['score_stats']['finite'] is True


def test_probe_config_supports_template_builders(tmp_path):
    config_path = tmp_path / 'probe_template.py'
    config_path.write_text(
        '\n'.join([
            "default_scope = 'baoiad'",
            "randomness = dict(seed=42, deterministic=False)",
            "train_dataloader = dict(",
            "    batch_size=2,",
            "    num_workers=0,",
            "    sampler=dict(type='DefaultSampler', shuffle=False),",
            "    dataset=dict(type='ProbeToyDataset', length=4, image_size=16),",
            ")",
            "test_dataloader = dict(",
            "    batch_size=2,",
            "    num_workers=0,",
            "    sampler=dict(type='DefaultSampler', shuffle=False),",
            "    dataset=dict(type='ProbeToyDataset', length=4, image_size=16),",
            ")",
            "val_dataloader = test_dataloader",
            "model = dict(type='ProbeTemplateBuilderDetector')",
        ]),
        encoding='utf-8',
    )

    report = probe_config(
        config_path=str(config_path),
        splits=('train', 'test'),
        max_batch_size=2,
        device='cpu',
    )

    assert report['passed'] is True
    assert report['splits']['test']['memory_bank_warmup']['used'] is True
    assert report['splits']['test']['memory_bank_warmup']['builder'] == 'build_template_from_dataloader'
    assert report['splits']['test']['predict']['score_stats']['finite'] is True


def test_probe_config_defaults_to_config_seed(tmp_path):
    config_path = tmp_path / 'probe_seed.py'
    config_path.write_text(
        '\n'.join([
            "default_scope = 'baoiad'",
            "randomness = dict(seed=123, deterministic=False)",
            "train_dataloader = dict(",
            "    batch_size=2,",
            "    num_workers=0,",
            "    sampler=dict(type='DefaultSampler', shuffle=False),",
            "    dataset=dict(type='ProbeToyDataset', length=4, image_size=16),",
            ")",
            "test_dataloader = dict(",
            "    batch_size=2,",
            "    num_workers=0,",
            "    sampler=dict(type='DefaultSampler', shuffle=False),",
            "    dataset=dict(type='ProbeToyDataset', length=4, image_size=16),",
            ")",
            "val_dataloader = test_dataloader",
            "model = dict(type='ProbeToyDetector')",
        ]),
        encoding='utf-8',
    )

    report = probe_config(
        config_path=str(config_path),
        splits=('train', 'test'),
        max_batch_size=2,
        device='cpu',
    )

    assert report['passed'] is True
    assert report['seed'] == 123


def test_probe_config_explicit_seed_overrides_config_seed(tmp_path):
    config_path = tmp_path / 'probe_seed_override.py'
    config_path.write_text(
        '\n'.join([
            "default_scope = 'baoiad'",
            "randomness = dict(seed=123, deterministic=False)",
            "train_dataloader = dict(",
            "    batch_size=2,",
            "    num_workers=0,",
            "    sampler=dict(type='DefaultSampler', shuffle=False),",
            "    dataset=dict(type='ProbeToyDataset', length=4, image_size=16),",
            ")",
            "test_dataloader = dict(",
            "    batch_size=2,",
            "    num_workers=0,",
            "    sampler=dict(type='DefaultSampler', shuffle=False),",
            "    dataset=dict(type='ProbeToyDataset', length=4, image_size=16),",
            ")",
            "val_dataloader = test_dataloader",
            "model = dict(type='ProbeToyDetector')",
        ]),
        encoding='utf-8',
    )

    report = probe_config(
        config_path=str(config_path),
        splits=('train', 'test'),
        max_batch_size=2,
        device='cpu',
        seed=7,
    )

    assert report['passed'] is True
    assert report['seed'] == 7


def test_prepare_dataloader_cfg_injects_default_single_class_target():
    cfg = _prepare_dataloader_cfg(
        dict(
            batch_size=4,
            num_workers=2,
            persistent_workers=True,
            dataset=dict(
                type='MVTecADDataset',
                data_root='data/mvtec_ad',
                split='test',
                multi_class=False,
                pipeline=[],
            ),
        ),
        max_batch_size=2,
    )

    assert cfg['batch_size'] == 2
    assert cfg['num_workers'] == 0
    assert cfg['persistent_workers'] is False
    assert cfg['dataset']['cls_names'] == ['bottle']

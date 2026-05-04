"""Tests for GanomalyDetector."""

from pathlib import Path
from unittest import TestCase

import torch
from mmengine import Config
from mmengine.optim import OptimWrapper, OptimWrapperDict

import baoiad  # noqa: F401
from baoiad.registry import MODELS
from baoiad.structures import ADDataSample

ROOT = Path(__file__).resolve().parents[3]


def _make_data_samples(batch_size, H=256, W=256):
    samples = []
    for i in range(batch_size):
        s = ADDataSample()
        s.gt_label = i % 2
        s.gt_mask = torch.zeros(H, W)
        s.cls_name = 'bottle'
        s.img_path = f'/fake/{i}.png'
        s.defect_type = 'good'
        samples.append(s)
    return samples

class TestGanomalyDetector(TestCase):
    def setUp(self):
        self.cfg = dict(type='GanomalyDetector', input_size=(64, 64), n_features=32, latent_vec_size=32, extra_layers=0)
        self.strict_cfg = dict(
            type='GanomalyDetector',
            strict=True,
            input_size=(64, 64),
            n_features=32,
            latent_vec_size=32,
            extra_layers=0,
        )

    def test_forward_tensor(self):
        model = MODELS.build(self.cfg)
        model.eval()
        out = model(torch.randn(2, 3, 64, 64), mode='tensor')
        assert out is not None

    def test_forward_loss(self):
        model = MODELS.build(self.cfg)
        model.train()
        data_samples = _make_data_samples(2, 64, 64)
        out = model(torch.randn(2, 3, 64, 64), data_samples, mode='loss')
        assert isinstance(out, dict)

    def test_forward_predict(self):
        model = MODELS.build(self.cfg)
        model.eval()
        data_samples = _make_data_samples(2, 64, 64)
        out = model(torch.randn(2, 3, 64, 64), data_samples, mode='predict')
        assert isinstance(out, list)
        assert len(out) == 2

    def test_strict_predict_uses_zero_placeholder_maps(self):
        model = MODELS.build(self.strict_cfg)
        model.eval()
        data_samples = _make_data_samples(2, 64, 64)
        out = model(torch.randn(2, 3, 64, 64), data_samples, mode='predict')

        assert len(out) == 2
        assert out[0].pred_anomaly_map.shape == (1, 64, 64)
        assert torch.count_nonzero(out[0].pred_anomaly_map) == 0
        assert torch.isfinite(out[0].pred_anomaly_map).all()

    def test_strict_train_step_with_split_optimizers(self):
        model = MODELS.build(self.strict_cfg)
        model.train()
        inputs = torch.randn(2, 3, 64, 64)
        data_samples = _make_data_samples(2, 64, 64)
        optim_wrapper = OptimWrapperDict(
            generator=OptimWrapper(torch.optim.Adam(model.generator.parameters(), lr=2e-4, betas=(0.5, 0.999))),
            discriminator=OptimWrapper(
                torch.optim.Adam(model.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
            ),
        )

        outputs = model.train_step(dict(inputs=inputs, data_samples=data_samples), optim_wrapper)

        assert sorted(outputs) == ['loss', 'loss_d', 'loss_g', 'loss_g_adv', 'loss_g_con', 'loss_g_enc']
        assert torch.isfinite(outputs['loss'])
        assert torch.isfinite(outputs['loss_g'])
        assert torch.isfinite(outputs['loss_d'])


def test_ganomaly_strict_config_freezes_image_only_protocol():
    cfg = Config.fromfile(ROOT / 'configs' / 'ganomaly' / 'ganomaly_256_mvtec_strict.py')

    assert cfg.benchmark_multi_class is False
    assert cfg.benchmark_keep_dataloader_workers is True
    assert cfg.benchmark_preserve_checkpoint_hooks is True
    assert cfg.benchmark_result_selector == {'mode': 'best', 'metric': 'image_auroc'}
    assert cfg.model.type == 'GanomalyDetector'
    assert cfg.model.strict is True
    assert tuple(cfg.model.input_size) == (256, 256)
    assert cfg.model.n_features == 64
    assert cfg.model.latent_vec_size == 100
    assert cfg.train_dataloader.batch_size == 64
    assert cfg.test_dataloader.batch_size == 64
    assert cfg.train_dataloader.num_workers == 8
    assert cfg.test_dataloader.num_workers == 8
    assert cfg.train_dataloader.drop_last is True
    assert cfg.train_dataloader.sampler.shuffle is True
    assert cfg.test_dataloader.sampler.shuffle is True
    train_pipeline = cfg.train_dataloader.dataset.pipeline
    test_pipeline = cfg.test_dataloader.dataset.pipeline
    assert train_pipeline[0].backend == 'pil'
    assert train_pipeline[1].backend == 'pil'
    assert train_pipeline[2].type == 'ResizeAD'
    assert train_pipeline[2].keep_ratio is True
    assert train_pipeline[2].backend == 'pillow'
    assert train_pipeline[2].official_pil is True
    assert train_pipeline[3].type == 'CenterCrop'
    assert train_pipeline[3].size == 256
    assert tuple(train_pipeline[4].mean) == (127.5, 127.5, 127.5)
    assert tuple(train_pipeline[4].std) == (127.5, 127.5, 127.5)
    assert test_pipeline[0].backend == 'pil'
    metric_cfg = cfg.val_evaluator.metrics[0]
    assert '_delete_' not in metric_cfg
    assert metric_cfg.type == 'AnomalyDetectionMetric'
    assert metric_cfg.metrics == ['image_auroc', 'image_f1max', 'image_ap', 'image_fpr@95tpr']
    assert 'pixel_auroc' not in metric_cfg.metrics
    assert metric_cfg.normalize_image_scores is True
    assert cfg.val_evaluator.get('type', None) is None
    assert cfg.optim_wrapper.constructor == 'GanomalyOptimWrapperConstructor'
    assert cfg.optim_wrapper.generator.optimizer.type == 'Adam'
    assert cfg.optim_wrapper.discriminator.optimizer.type == 'Adam'
    assert cfg.optim_wrapper.generator.optimizer.lr == 2e-4
    assert tuple(cfg.optim_wrapper.generator.optimizer.betas) == (0.5, 0.999)
    assert cfg.default_hooks.checkpoint.save_best == 'ad/image_auroc'
    assert cfg.train_cfg.max_epochs == 15
    assert cfg.train_cfg.val_interval == 1

"""Tests for RealNetDetector."""

import os
from types import SimpleNamespace
from unittest import TestCase

import torch
import torch.nn.functional as F
from mmengine import Config

from baoiad.engine.hooks.realnet_init_hook import RealNetInitHook
from baoiad.models.detectors.realnet import ReconstructionUNet, SimpleUNet
from baoiad.registry import MODELS
from baoiad.structures import ADDataSample

import baoiad  # noqa: F401


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _make_train_samples(batch_size, height=64, width=64):
    samples = []
    for i in range(batch_size):
        sample = ADDataSample()
        sample.set_metainfo({
            'cls_name': 'bottle',
            'img_path': f'/fake/{i}.png',
            'defect_type': 'good',
            'clean_img': torch.randn(3, height, width),
            'anomaly_type': 'sdas',
        })
        mask = torch.zeros(height, width)
        mask[height // 4: height // 2, width // 4: width // 2] = 1.0
        sample.gt_mask = mask
        sample.gt_label = 0
        samples.append(sample)
    return samples


def _make_test_samples(batch_size, height=64, width=64):
    samples = []
    for i in range(batch_size):
        sample = ADDataSample()
        sample.set_metainfo({
            'cls_name': 'bottle',
            'img_path': f'/fake/{i}.png',
            'defect_type': 'good',
        })
        sample.gt_mask = torch.zeros(height, width)
        sample.gt_label = i % 2
        samples.append(sample)
    return samples


class _AFSLoader:
    def __init__(self, batches, anomaly_types):
        self._batches = list(batches)
        self.dataset = SimpleNamespace(anomaly_types=dict(anomaly_types))
        self.snapshots = []

    def __iter__(self):
        for batch in self._batches:
            self.snapshots.append(dict(self.dataset.anomaly_types))
            yield batch


class TestRealNetDetector(TestCase):
    def setUp(self):
        self.cfg = dict(
            type='RealNetDetector',
            backbone=dict(
                type='TIMMBackbone',
                model_name='wide_resnet50_2',
                pretrained=False,
                features_only=True,
                out_indices=(1,),
                frozen=True,
            ),
            structure=[
                dict(name='block1', layers=[dict(idx='layer1', planes=16)], stride=4),
            ],
            init_bsn=2,
            reconstruction_type='official',
            num_res_blocks=1,
            hide_channels_ratio=0.5,
            channel_mult=[1, 2],
            attention_mult=[2],
            num_residual_layers=1,
            rrs_modes=['max', 'mean'],
            rrs_mode_numbers=[8, 8],
            image_score_pool_size=(4, 4),
            dtd_path=None,
        )

    def test_forward_tensor(self):
        model = MODELS.build(self.cfg)
        model.eval()
        out = model(torch.randn(2, 3, 64, 64), mode='tensor')
        assert isinstance(out, dict)
        assert 'selected_feats' in out
        assert 'recon_feats' in out
        assert 'residuals' in out
        assert 'logit_mask' in out
        assert 'pred_map' in out
        assert 'img_scores' in out

    def test_forward_loss(self):
        model = MODELS.build(self.cfg)
        model.train()
        data_samples = _make_train_samples(2)
        out = model(torch.randn(2, 3, 64, 64), data_samples, mode='loss')
        assert isinstance(out, dict)
        assert 'loss' in out
        assert torch.isfinite(out['loss'])
        assert torch.isfinite(out['loss_seg'])
        assert torch.isfinite(out['loss_feat'])

    def test_forward_predict(self):
        model = MODELS.build(self.cfg)
        model.eval()
        data_samples = _make_test_samples(2)
        out = model(torch.randn(2, 3, 64, 64), data_samples, mode='predict')
        assert isinstance(out, list)
        assert len(out) == 2
        for sample in out:
            assert hasattr(sample, 'pred_score')
            assert hasattr(sample, 'pred_anomaly_map')
            assert sample.pred_anomaly_map.shape == (1, 64, 64)

    def test_afs_selection(self):
        model = MODELS.build(self.cfg)
        for name in model.layer_names:
            indices = model.afs_indices[name]
            assert indices.shape[0] == model.selected_channels[name]

    def test_reconstruction_modules_use_official_path_by_default(self):
        model = MODELS.build(self.cfg)
        recon = model.recon_modules['block1']
        assert isinstance(recon, ReconstructionUNet)
        assert not isinstance(recon, SimpleUNet)

    def test_image_score_uses_avgpool_path(self):
        model = MODELS.build(self.cfg)
        pred_map = torch.arange(1, 26, dtype=torch.float32).view(1, 1, 5, 5)
        expected = F.avg_pool2d(pred_map, (4, 4), stride=1).reshape(1, -1).max(dim=1).values
        actual = model._image_scores_from_map(pred_map)
        assert torch.allclose(actual, expected)

    def test_predict_invert_map_flips_output_map(self):
        base_model = MODELS.build(self.cfg)
        inv_cfg = dict(self.cfg)
        inv_cfg['predict_invert_map'] = True
        inv_model = MODELS.build(inv_cfg)
        inv_model.load_state_dict(base_model.state_dict())

        inputs = torch.randn(2, 3, 64, 64)
        data_samples = _make_test_samples(2)

        base_out = base_model(inputs, data_samples, mode='predict')
        inv_out = inv_model(inputs, _make_test_samples(2), mode='predict')

        for base_sample, inv_sample in zip(base_out, inv_out):
            assert torch.allclose(
                inv_sample.pred_anomaly_map,
                1.0 - base_sample.pred_anomaly_map,
                atol=1e-5,
            )

    def test_init_afs_updates_indices(self):
        model = MODELS.build(self.cfg)
        default_indices = model.afs_indices['block1'].clone()
        loader = [{
            'inputs': torch.randn(2, 3, 64, 64),
            'data_samples': _make_train_samples(2),
        }]
        model.init_afs(loader)
        updated = model.afs_indices['block1']
        assert model.afs_initialized is True
        assert updated.shape == default_indices.shape
        assert not torch.equal(updated, default_indices)

    def test_init_afs_uses_only_anomaly_sampling_and_restores_dataset_state(self):
        model = MODELS.build(self.cfg)
        loader = _AFSLoader(
            batches=[{
                'inputs': torch.randn(2, 3, 64, 64),
                'data_samples': _make_train_samples(2),
            }],
            anomaly_types={'normal': 0.5, 'sdas': 0.3, 'dtd': 0.2},
        )
        model.init_afs(loader)
        assert loader.dataset.anomaly_types == {'normal': 0.5, 'sdas': 0.3, 'dtd': 0.2}
        assert loader.snapshots
        for snapshot in loader.snapshots:
            assert 'normal' not in snapshot
            assert snapshot == {'sdas': 0.6, 'dtd': 0.4}


def test_realnet_init_hook_only_runs_once():
    class _DummyModel:
        def __init__(self):
            self.afs_initialized = False
            self.calls = 0

        def init_afs(self, _train_dataloader):
            self.calls += 1
            self.afs_initialized = True

    class _DummyLogger:
        def info(self, *_args, **_kwargs):
            return None

    runner = SimpleNamespace(
        model=_DummyModel(),
        train_dataloader=[],
        logger=_DummyLogger(),
    )
    hook = RealNetInitHook()
    hook.before_train(runner)
    hook.before_train(runner)
    assert runner.model.calls == 1


def test_realnet_strict_config_uses_official_seed_and_raw_map_semantics():
    cfg = Config.fromfile(os.path.join(ROOT, 'configs', 'realnet', 'realnet_wrn50_256_mvtec_strict.py'))
    assert cfg.randomness.seed == 100
    assert cfg.model.anomaly_channel_index == 1
    assert cfg.model.predict_invert_map is False
    assert cfg.test_evaluator.flip_auroc_if_below_half is True


def test_anomaly_channel_index_zero_flips_training_targets_and_predict_map():
    cfg = dict(
        type='RealNetDetector',
        backbone=dict(
            type='TIMMBackbone',
            model_name='wide_resnet50_2',
            pretrained=False,
            features_only=True,
            out_indices=(1,),
            frozen=True,
        ),
        structure=[dict(name='block1', layers=[dict(idx='layer1', planes=16)], stride=4)],
        init_bsn=2,
        reconstruction_type='official',
        num_res_blocks=1,
        hide_channels_ratio=0.5,
        channel_mult=[1, 2],
        attention_mult=[2],
        num_residual_layers=1,
        rrs_modes=['max', 'mean'],
        rrs_mode_numbers=[8, 8],
        image_score_pool_size=(4, 4),
        anomaly_channel_index=0,
        dtd_path=None,
    )
    model = MODELS.build(cfg)
    model.eval()
    inputs = torch.randn(2, 3, 64, 64)
    outputs = model(inputs, mode='tensor')
    expected = torch.softmax(outputs['logit_mask'], dim=1)[:, :1]
    assert torch.allclose(outputs['pred_map'], expected, atol=1e-6)

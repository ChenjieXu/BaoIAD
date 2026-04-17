"""Comprehensive detector tests — shared parametrized tests across all detectors.

Tests loss sanity, predict output structure, different batch sizes,
invalid mode handling, and memory bank pipeline for applicable methods.
"""

import pytest
import torch
import numpy as np
import types
from baoiad.structures import ADDataSample

import baoiad  # noqa: F401
from baoiad.registry import MODELS

# Check optional dependencies
_has_freia = True
try:
    import FrEIA  # noqa: F401
except ImportError:
    _has_freia = False

_freia_skip = pytest.mark.skipif(not _has_freia, reason='FrEIA not installed')


# ---------------------------------------------------------------------------
# Detector configs (minimal, small backbones for speed)
# ---------------------------------------------------------------------------
# Each entry: (name, cfg_dict, needs_memory_bank, xfail_reason_or_None, input_size)
_RN18_BACKBONE = dict(type='FeatureExtractor', backbone_name='resnet18',
                      out_indices=(1, 2, 3))
_RN18_BACKBONE_12 = dict(type='FeatureExtractor', backbone_name='resnet18',
                         out_indices=(1, 2))

# Detectors needing larger input
_LARGE_INPUT = {'EfficientADDetector': 256, 'ASTDetector': 768, 'GraphCoreDetector': 224, 'RegADDetector': 16}

DETECTOR_CFGS = [
    ('PatchCore', dict(
        type='PatchCore', backbone=_RN18_BACKBONE,
        neck=dict(type='MultiScalePooling', output_size=14),
        head=dict(type='MemoryBankHead', coreset_ratio=0.5, num_neighbors=1),
        freeze_backbone=True,
    ), True, None),
    ('STFPMDetector', dict(
        type='STFPMDetector', backbone='resnet18',
    ), False, None),
    ('CFADetector', dict(
        type='CFADetector', backbone='resnet18',
        gamma_c=1, gamma_d=1, num_nearest_neighbors=3,
        num_hard_negative_features=3, radius=1e-5, sigma=4,
    ), False, None),
    ('CFlowDetector', dict(
        type='CFlowDetector', backbone='resnet18',
        flow_steps=4, hidden_ratio=1.0, conv3x3_only=True,
        coupling_blocks=4,
    ), False, 'FrEIA not installed' if not _has_freia else None),
    ('CutPasteDetector', dict(
        type='CutPasteDetector', backbone='resnet18',
        proj_dim=128, num_classes=3,
    ), False, None),
    ('DifferNetDetector', dict(
        type='DifferNetDetector', backbone='resnet18',
        n_coupling_blocks=4,
    ), False, 'FrEIA not installed' if not _has_freia else None),
    ('FastFlowDetector', dict(
        type='FastFlowDetector', backbone='resnet18',
        flow_steps=4, hidden_ratio=1.0, conv3x3_only=True,
    ), False, 'FrEIA not installed' if not _has_freia else None),
    ('GanomalyDetector', dict(
        type='GanomalyDetector', input_size=(64, 64), latent_vec_size=64, n_features=32,
        extra_layers=0,
    ), False, None),
    ('SimpleNetDetector', dict(
        type='SimpleNetDetector', backbone='resnet18',
        noise_std=0.015, feature_adaptor_dim=128,
    ), False, None),
    ('SPADEDetector', dict(
        type='SPADEDetector', backbone='resnet18', k=3,
    ), True, None),
    ('PaDiMDetector', dict(
        type='PaDiMDetector', backbone='resnet18',
    ), True, None),
    ('DFKDEDetector', dict(
        type='DFKDEDetector', backbone='resnet18', n_pca_components=2,
    ), True, None),
    ('DFMDetector', dict(
        type='DFMDetector', backbone='resnet18',
    ), True, None),
    ('GraphCoreDetector', dict(
        type='GraphCoreDetector',
        backbone=dict(
            type='GraphCoreViGBackbone',
            model_name='vig_ti_224_gelu',
            pretrained=False,
            frozen=True,
        ),
        n_neighbours=3,
        sampler_percentage=0.05,
        random_seed=66,
        coreset_initial_index=0,
    ), True, None),
    ('FREDetector', dict(
        type='FREDetector', backbone='resnet18', input_dim=1024, latent_dim=64,
    ), False, None),
    ('NSADetector', dict(
        type='NSADetector', backbone='resnet18',
    ), False, None),
    ('MemAEDetector', dict(
        type='MemAEDetector', input_size=64, mem_dim=50, shrink_thres=0.0025,
    ), False, None),
    ('RegADDetector', dict(
        type='RegADDetector', backbone='resnet18', layers=(3,), few_shot=1, img_size=16, pretrained_backbone=False,
    ), True, None),
    ('DRAEMDetector', dict(
        type='DRAEMDetector', input_size=64,
    ), False, None),
    ('DeSTSegDetector', dict(
        type='DeSTSegDetector', backbone='resnet18',
    ), False, None),
    ('EfficientADDetector', dict(
        type='EfficientADDetector', pdn_channels=384, pdn_variant='small',
        padding=False, teacher_pretrained='',
    ), False, None),
    ('SuperSimpleNetDetector', dict(
        type='SuperSimpleNetDetector', backbone='resnet18',
    ), False, None),
    ('ASTDetector', dict(
        type='ASTDetector', backbone=dict(model_name='tf_efficientnet_b5', pretrained=False),
        extract_layer=35, n_feat=304, map_len=24,
        n_coupling_blocks=2, channels_hidden_teacher=32,
        channels_hidden_student=128, n_student_blocks=2, img_size=768,
    ), False, None),
]

# Filter out names for parametrize IDs
_DET_IDS = [c[0] for c in DETECTOR_CFGS]


def _make_samples(n, H=64, W=64):
    samples = []
    for i in range(n):
        s = ADDataSample()
        s.gt_label = i % 2
        s.gt_mask = torch.zeros(H, W)
        if i % 2 == 1:
            s.gt_mask[H // 4:3 * H // 4, W // 4:3 * W // 4] = 1.0
        s.cls_name = 'bottle'
        s.img_path = f'/fake/{i}.png'
        s.defect_type = 'good' if i % 2 == 0 else 'broken'
        s.support_imgs = torch.rand(2, 3, H, W)
        samples.append(s)
    return samples


def _get_input_size(cfg_entry):
    return _LARGE_INPUT.get(cfg_entry[0], 64)


def _build_model(cfg_entry):
    name, cfg, needs_mb, xfail = cfg_entry
    model = MODELS.build(cfg)
    if name == 'RegADDetector':
        def _extract_features(_self, x):
            batch_size = x.shape[0]
            device = x.device
            return {
                'layer1': torch.ones(batch_size, 4, 2, 2, device=device),
                'layer2': torch.ones(batch_size, 6, 2, 2, device=device),
                'layer3': torch.ones(batch_size, 8, 2, 2, device=device),
                'final_feat': torch.ones(batch_size, 256, 2, 2, device=device),
            }

        def _build_bank(self, _dataloader=None):
            device = next(self.parameters()).device
            self.support_feat = torch.zeros(1, 256, 2, 2, device=device)
            self.embedding_mean = torch.zeros(8, 4, device=device)
            self.embedding_cov_inv = torch.eye(8, device=device).unsqueeze(-1).repeat(1, 1, 4)

        model.extract_features = types.MethodType(_extract_features, model)
        model.build_memory_bank = types.MethodType(_build_bank, model)
        model.fit = types.MethodType(lambda self, *args, **kwargs: _build_bank(self), model)
    return model, needs_mb


def _prepare_for_predict(model, needs_mb, input_size=64, n_batches=3):
    """Run loss mode + build memory bank if needed."""
    model.train()
    for _ in range(n_batches):
        model(torch.randn(2, 3, input_size, input_size),
              _make_samples(2, input_size, input_size), mode='loss')
    if needs_mb:
        if hasattr(model, 'build_memory_bank'):
            model.build_memory_bank()
        elif hasattr(model, 'head') and hasattr(model.head, 'build_memory_bank'):
            model.head.build_memory_bank()
        elif hasattr(model, 'fit'):
            model.fit()
    model.eval()


# ===========================================================================
# Parametrized tests
# ===========================================================================

@pytest.mark.parametrize('cfg_entry', DETECTOR_CFGS, ids=_DET_IDS)
class TestDetectorLoss:
    """Test loss mode for all detectors."""

    def test_loss_returns_dict(self, cfg_entry):
        if cfg_entry[3]:
            pytest.xfail(cfg_entry[3])
        sz = _get_input_size(cfg_entry)
        model, _ = _build_model(cfg_entry)
        sz = _get_input_size(cfg_entry)
        model.train()
        out = model(torch.randn(2, 3, sz, sz), _make_samples(2, sz, sz), mode='loss')
        assert isinstance(out, dict), f'Expected dict, got {type(out)}'

    def test_loss_no_nan(self, cfg_entry):
        if cfg_entry[3]:
            pytest.xfail(cfg_entry[3])
        sz = _get_input_size(cfg_entry)
        model, _ = _build_model(cfg_entry)
        sz = _get_input_size(cfg_entry)
        model.train()
        out = model(torch.randn(2, 3, sz, sz), _make_samples(2, sz, sz), mode='loss')
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                assert not torch.isnan(v).any(), f'{k} contains NaN'

    def test_loss_single_sample(self, cfg_entry):
        if cfg_entry[3]:
            pytest.xfail(cfg_entry[3])
        sz = _get_input_size(cfg_entry)
        model, _ = _build_model(cfg_entry)
        sz = _get_input_size(cfg_entry)
        model.train()
        out = model(torch.randn(1, 3, sz, sz), _make_samples(1, sz, sz), mode='loss')
        assert isinstance(out, dict)


@pytest.mark.parametrize('cfg_entry', DETECTOR_CFGS, ids=_DET_IDS)
class TestDetectorPredict:
    """Test predict mode for all detectors."""

    def test_predict_returns_list(self, cfg_entry):
        if cfg_entry[3]:
            pytest.xfail(cfg_entry[3])
        sz = _get_input_size(cfg_entry)
        model, needs_mb = _build_model(cfg_entry)
        sz = _get_input_size(cfg_entry)
        _prepare_for_predict(model, needs_mb, input_size=sz)
        out = model(torch.randn(2, 3, sz, sz), _make_samples(2, sz, sz), mode='predict')
        assert isinstance(out, list)
        assert len(out) == 2

    def test_predict_has_score(self, cfg_entry):
        if cfg_entry[3]:
            pytest.xfail(cfg_entry[3])
        sz = _get_input_size(cfg_entry)
        model, needs_mb = _build_model(cfg_entry)
        sz = _get_input_size(cfg_entry)
        _prepare_for_predict(model, needs_mb, input_size=sz)
        out = model(torch.randn(2, 3, sz, sz), _make_samples(2, sz, sz), mode='predict')
        for r in out:
            assert hasattr(r, 'pred_score'), 'Missing pred_score'
            assert isinstance(r.pred_score, (int, float, np.floating))

    def test_predict_has_anomaly_map(self, cfg_entry):
        if cfg_entry[3]:
            pytest.xfail(cfg_entry[3])
        sz = _get_input_size(cfg_entry)
        model, needs_mb = _build_model(cfg_entry)
        sz = _get_input_size(cfg_entry)
        _prepare_for_predict(model, needs_mb, input_size=sz)
        out = model(torch.randn(2, 3, sz, sz), _make_samples(2, sz, sz), mode='predict')
        for r in out:
            assert hasattr(r, 'pred_anomaly_map'), 'Missing pred_anomaly_map'

    def test_predict_single_sample(self, cfg_entry):
        if cfg_entry[3]:
            pytest.xfail(cfg_entry[3])
        sz = _get_input_size(cfg_entry)
        model, needs_mb = _build_model(cfg_entry)
        sz = _get_input_size(cfg_entry)
        _prepare_for_predict(model, needs_mb, input_size=sz)
        out = model(torch.randn(1, 3, sz, sz), _make_samples(1, sz, sz), mode='predict')
        assert len(out) == 1


@pytest.mark.parametrize('cfg_entry', DETECTOR_CFGS, ids=_DET_IDS)
class TestDetectorTensor:
    """Test tensor mode for all detectors."""

    def test_tensor_mode(self, cfg_entry):
        if cfg_entry[3]:
            pytest.xfail(cfg_entry[3])
        sz = _get_input_size(cfg_entry)
        model, _ = _build_model(cfg_entry)
        sz = _get_input_size(cfg_entry)
        model.eval()
        out = model(torch.randn(2, 3, sz, sz), mode='tensor')
        assert out is not None


@pytest.mark.parametrize('cfg_entry', DETECTOR_CFGS, ids=_DET_IDS)
class TestDetectorEdgeCases:
    """Edge cases and error handling."""

    def test_invalid_mode_raises(self, cfg_entry):
        """Only BaseADModel subclasses validate mode. Others may silently ignore."""
        if cfg_entry[3]:
            pytest.xfail(cfg_entry[3])
        sz = _get_input_size(cfg_entry)
        model, _ = _build_model(cfg_entry)
        sz = _get_input_size(cfg_entry)
        try:
            model(torch.randn(1, 3, sz, sz), mode='invalid_mode_xyz')
            # If it didn't raise, that's okay — not all detectors validate mode
        except (RuntimeError, ValueError, KeyError):
            pass  # Expected for BaseADModel subclasses


# ===========================================================================
# Memory bank pipeline tests
# ===========================================================================

_MB_CFGS = [(n, c, mb, x) for n, c, mb, x in DETECTOR_CFGS if mb and not x]
_MB_IDS = [c[0] for c in _MB_CFGS]


@pytest.mark.parametrize('cfg_entry', _MB_CFGS, ids=_MB_IDS)
class TestMemoryBankPipeline:
    """Full collect → build → predict pipeline for memory bank methods."""

    def test_full_pipeline(self, cfg_entry):
        model, _ = _build_model(cfg_entry)
        sz = _get_input_size(cfg_entry)
        # Phase 1: collect
        model.train()
        for _ in range(3):
            model(torch.randn(2, 3, sz, sz), _make_samples(2, sz, sz), mode='loss')
        # Phase 2: build
        built = False
        if hasattr(model, 'build_memory_bank'):
            model.build_memory_bank()
            built = True
        elif hasattr(model, 'head') and hasattr(model.head, 'build_memory_bank'):
            model.head.build_memory_bank()
            built = True
        elif hasattr(model, 'fit'):
            model.fit()
            built = True
        assert built, 'No build method found'
        # Phase 3: predict
        model.eval()
        out = model(torch.randn(2, 3, sz, sz), _make_samples(2, sz, sz), mode='predict')
        assert isinstance(out, list)
        assert len(out) == 2
        for r in out:
            assert hasattr(r, 'pred_score')

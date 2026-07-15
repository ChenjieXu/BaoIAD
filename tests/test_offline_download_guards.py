"""Regression tests for BaoIAD-managed downloads in explicit offline mode."""

from __future__ import annotations

import pytest
import torch

import baoiad.runtime as runtime_module
from baoiad.runtime import OfflineModeError


@pytest.fixture(autouse=True)
def _offline_without_network(monkeypatch):
    monkeypatch.setenv('BAOIAD_OFFLINE', '1')

    def unexpected_download(*args, **kwargs):
        raise AssertionError('the downloader was called before the offline guard')

    monkeypatch.setattr(runtime_module, 'urlopen', unexpected_download)


def test_dtd_download_is_rejected_before_network(tmp_path):
    from baoiad.utils.dtd import download_dtd

    with pytest.raises(OfflineModeError, match='DTD texture dataset'):
        download_dtd(str(tmp_path / 'dtd'))


def test_efficientad_downloads_are_rejected_before_network(tmp_path):
    from baoiad.models.detectors.efficientad import (
        _download_imagenette,
        _download_pretrained_weights,
    )

    with pytest.raises(OfflineModeError, match='EfficientAD teacher weights'):
        _download_pretrained_weights(str(tmp_path / 'weights'))
    with pytest.raises(OfflineModeError, match='ImageNette dataset'):
        _download_imagenette(str(tmp_path / 'imagenette'))


def test_dsr_download_is_rejected_before_network(tmp_path):
    from baoiad.models.detectors.dsr import _download_vqvae_weights

    with pytest.raises(OfflineModeError, match='DSR VQ-VAE weights'):
        _download_vqvae_weights(str(tmp_path / 'weights'))


def test_saa_downloads_are_rejected_before_network(monkeypatch, tmp_path):
    from baoiad.models.detectors import saa

    monkeypatch.setattr(saa, '_WEIGHTS_DIR', str(tmp_path / 'weights'))
    with pytest.raises(OfflineModeError, match='GroundingDINO weights'):
        saa._download_gdino_weights()
    with pytest.raises(OfflineModeError, match='SAM weights'):
        saa._download_sam_weights()


def test_destseg_download_is_rejected_before_network(monkeypatch, tmp_path):
    from baoiad.models.detectors import destseg

    monkeypatch.setattr(destseg, '_LEGACY_RESNET18_PATH', str(tmp_path / 'teacher.pth'))
    with pytest.raises(OfflineModeError, match='DeSTSeg teacher checkpoint'):
        destseg.DeSTSegDetector._ensure_legacy_teacher_checkpoint()


def test_dinomaly_download_is_rejected_before_network(tmp_path):
    from baoiad.models.backbones.dinomaly_backbone import _download_and_load_weights

    with pytest.raises(OfflineModeError, match='DINOv2 weights'):
        _download_and_load_weights(
            torch.nn.Identity(),
            model_type='dinov2',
            arch='base',
            patch_size=14,
            cache_dir=str(tmp_path),
        )


def test_vitad_cached_url_guard_runs_before_torch_hub(monkeypatch, tmp_path):
    from baoiad.models.backbones.vitad_backbone import DistilledVisionTransformerBackbone

    monkeypatch.setattr(torch.hub, 'get_dir', lambda: str(tmp_path / 'hub'))
    monkeypatch.setattr(
        torch.hub,
        'load_state_dict_from_url',
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError('torch.hub was called before the offline guard')
        ),
    )
    with pytest.raises(OfflineModeError, match='ViTAD backbone weights'):
        DistilledVisionTransformerBackbone(
            teachers=(1,),
            neck=(),
            img_size=32,
            patch_size=16,
            embed_dim=8,
            depth=1,
            num_heads=1,
            mlp_ratio=1,
            pretrained_url='https://example.invalid/model.pth',
        )


def test_musc_torch_hub_repo_is_rejected_before_network(monkeypatch, tmp_path):
    from baoiad.models.backbones.musc_clip_backbone import MuScDINOv2Backbone

    monkeypatch.setattr(torch.hub, 'get_dir', lambda: str(tmp_path / 'hub'))
    monkeypatch.setattr(
        torch.hub,
        'load',
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError('torch.hub was called before the offline guard')
        ),
    )
    with pytest.raises(OfflineModeError, match='DINOv2 torch.hub repository'):
        MuScDINOv2Backbone()


def test_draem_does_not_downgrade_offline_error(monkeypatch, tmp_path):
    from baoiad.datasets import draem_dataset

    monkeypatch.setattr(
        draem_dataset,
        '_download_dtd',
        lambda: (_ for _ in ()).throw(OfflineModeError('offline DTD')),
    )
    with pytest.raises(OfflineModeError, match='offline DTD'):
        draem_dataset.DRAEMDataset(data_root=str(tmp_path), dtd_path='auto')


def test_glass_does_not_downgrade_offline_error(monkeypatch):
    from baoiad.models.detectors import glass

    monkeypatch.setattr(
        glass,
        'resolve_dtd_texture_paths',
        lambda path: (_ for _ in ()).throw(OfflineModeError('offline DTD')),
    )
    detector = glass.GLASSDetector.__new__(glass.GLASSDetector)
    with pytest.raises(OfflineModeError, match='offline DTD'):
        detector._load_dtd_textures('auto')


def test_memseg_does_not_downgrade_offline_error(monkeypatch):
    from baoiad.models.detectors import memseg

    monkeypatch.setattr(
        memseg,
        '_download_dtd',
        lambda: (_ for _ in ()).throw(OfflineModeError('offline DTD')),
    )
    detector = memseg.MemSegDetector.__new__(memseg.MemSegDetector)
    detector._dtd_dir = None
    detector.dtd_path = 'auto'
    detector.require_texture_source = False
    with pytest.raises(OfflineModeError, match='offline DTD'):
        detector._get_dtd_dir()


def test_efficientad_does_not_use_random_teacher_offline(monkeypatch, tmp_path):
    from baoiad.models.detectors import efficientad

    monkeypatch.setattr(efficientad, '_WEIGHTS_DIR', str(tmp_path / 'weights'))
    monkeypatch.setattr(
        efficientad,
        '_download_pretrained_weights',
        lambda: (_ for _ in ()).throw(OfflineModeError('offline teacher')),
    )
    detector = efficientad.EfficientADDetector.__new__(efficientad.EfficientADDetector)
    with pytest.raises(OfflineModeError, match='offline teacher'):
        detector._load_teacher_weights('auto', 'small')


def test_dsr_does_not_use_random_weights_offline(monkeypatch, tmp_path):
    from baoiad.models.detectors import dsr

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        dsr,
        '_download_vqvae_weights',
        lambda *args, **kwargs: (_ for _ in ()).throw(OfflineModeError('offline VQ-VAE')),
    )
    with pytest.raises(OfflineModeError, match='offline VQ-VAE'):
        dsr.DSRDetector._resolve_vqvae_path('auto')

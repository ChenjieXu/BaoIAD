"""Tests for explicit offline/network policy."""

from __future__ import annotations

import os
from io import BytesIO
from types import SimpleNamespace

import pytest
import torch

from baoiad.runtime import (
    OfflineModeError,
    configure_offline_mode,
    download_url,
    is_offline_mode,
    require_network,
    require_torchvision_weights,
)


def test_disabled_offline_mode_preserves_user_environment():
    environ = {'HF_HUB_OFFLINE': '0', 'CUSTOM': 'value'}

    configure_offline_mode(False, environ=environ)

    assert environ == {'HF_HUB_OFFLINE': '0', 'CUSTOM': 'value'}


def test_explicit_offline_mode_configures_supported_hubs():
    environ = {}

    configure_offline_mode(True, environ=environ)

    assert environ == {
        'BAOIAD_OFFLINE': '1',
        'HF_HUB_OFFLINE': '1',
        'TRANSFORMERS_OFFLINE': '1',
        'HF_DATASETS_OFFLINE': '1',
    }
    assert is_offline_mode(environ)


@pytest.mark.parametrize('value', ['1', 'true', 'YES', 'on'])
def test_user_offline_environment_is_respected(value):
    assert is_offline_mode({'HF_HUB_OFFLINE': value})


def test_network_guard_is_noop_online(monkeypatch):
    for name in ('BAOIAD_OFFLINE', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
        monkeypatch.delenv(name, raising=False)

    require_network('download a fixture')


def test_network_guard_fails_before_download(monkeypatch):
    monkeypatch.setenv('BAOIAD_OFFLINE', '1')

    with pytest.raises(OfflineModeError, match='Provide the required file'):
        require_network('download a fixture', url='https://example.invalid/file')


def test_runtime_module_has_no_import_side_effect(monkeypatch):
    before = dict(os.environ)
    configure_offline_mode(False)
    assert dict(os.environ) == before


def test_torchvision_offline_requires_cached_weights(monkeypatch, tmp_path):
    monkeypatch.setenv('BAOIAD_OFFLINE', '1')
    monkeypatch.setattr(torch.hub, 'get_dir', lambda: str(tmp_path / 'hub'))
    weights = SimpleNamespace(url='https://example.invalid/resnet.pth')

    with pytest.raises(OfflineModeError, match='resnet.pth'):
        require_torchvision_weights(weights, action='load ResNet weights')


def test_torchvision_offline_accepts_cached_weights(monkeypatch, tmp_path):
    monkeypatch.setenv('BAOIAD_OFFLINE', '1')
    monkeypatch.setattr(torch.hub, 'get_dir', lambda: str(tmp_path / 'hub'))
    checkpoint = tmp_path / 'hub' / 'checkpoints' / 'resnet.pth'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'cached')

    require_torchvision_weights(
        SimpleNamespace(url='https://example.invalid/resnet.pth'),
        action='load ResNet weights',
    )


def test_download_url_uses_scoped_timeout_and_atomic_destination(monkeypatch, tmp_path):
    for name in ('BAOIAD_OFFLINE', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
        monkeypatch.delenv(name, raising=False)
    calls = []

    def fake_urlopen(url, *, timeout):
        calls.append((url, timeout))
        return BytesIO(b'weights')

    monkeypatch.setattr('baoiad.runtime.urlopen', fake_urlopen)
    destination = tmp_path / 'cache' / 'model.pth'

    result = download_url(
        'https://example.invalid/model.pth',
        destination,
        action='download test weights',
        timeout=17,
    )

    assert result == str(destination.resolve())
    assert destination.read_bytes() == b'weights'
    assert not (destination.parent / 'model.pth.part').exists()
    assert calls == [('https://example.invalid/model.pth', 17)]

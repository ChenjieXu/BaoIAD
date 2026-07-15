"""Security regression tests for BaoIAD checkpoint loading."""

from __future__ import annotations

import os
import pickle
import threading
import warnings
from pathlib import Path

import pytest
import torch
from mmengine.logging import HistoryBuffer, MessageHub
from mmengine.runner.checkpoint import CheckpointLoader

from baoiad.checkpoint import (
    CheckpointLoadError,
    TrustedCheckpointWarning,
    UnsafeCheckpointError,
    checkpoint_loading_policy,
    load_checkpoint,
    trusted_checkpoint_loading_enabled,
)


def _write_marker(marker_path: str) -> None:
    Path(marker_path).write_text('executed', encoding='utf-8')


class _MaliciousPayload:
    def __init__(self, marker_path: Path) -> None:
        self.marker_path = marker_path

    def __reduce__(self):
        return _write_marker, (str(self.marker_path),)


def _save_malicious_checkpoint(checkpoint_path: Path, marker_path: Path) -> None:
    torch.save(
        {'state_dict': {'weight': torch.ones(1)}, 'payload': _MaliciousPayload(marker_path)},
        checkpoint_path,
    )


def test_default_loader_accepts_tensor_only_state_dict(tmp_path):
    checkpoint_path = tmp_path / 'tensor-only.pth'
    expected = {
        'state_dict': {
            'weight': torch.arange(6, dtype=torch.float32).reshape(2, 3),
            'bias': torch.zeros(2),
        },
        'meta': {'epoch': 3},
    }
    torch.save(expected, checkpoint_path)

    checkpoint = load_checkpoint(checkpoint_path)

    assert checkpoint['meta'] == {'epoch': 3}
    assert torch.equal(
        checkpoint['state_dict']['weight'], expected['state_dict']['weight'])
    assert torch.equal(
        checkpoint['state_dict']['bias'], expected['state_dict']['bias'])


def test_corrupt_checkpoint_is_an_operational_load_error(tmp_path):
    checkpoint_path = tmp_path / 'corrupt.pth'
    checkpoint_path.write_bytes(b'not a valid torch checkpoint')

    with pytest.raises(CheckpointLoadError) as exc_info:
        load_checkpoint(checkpoint_path)

    message = str(exc_info.value)
    assert str(checkpoint_path) in message
    assert isinstance(exc_info.value.__cause__, pickle.UnpicklingError)
    assert str(exc_info.value.__cause__) in message
    assert '--trusted-checkpoint' not in message


def test_invalid_map_location_is_an_operational_load_error(tmp_path):
    checkpoint_path = tmp_path / 'tensor-only.pth'
    torch.save({'state_dict': {'weight': torch.ones(1)}}, checkpoint_path)

    with pytest.raises(CheckpointLoadError) as exc_info:
        load_checkpoint(
            checkpoint_path, map_location='definitely-not-a-device')

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) in str(exc_info.value)
    assert '--trusted-checkpoint' not in str(exc_info.value)


def test_default_loader_rejects_malicious_pickle_without_executing(tmp_path):
    checkpoint_path = tmp_path / 'untrusted.pth'
    marker_path = tmp_path / 'executed.txt'
    _save_malicious_checkpoint(checkpoint_path, marker_path)

    with pytest.raises(UnsafeCheckpointError, match='--trusted-checkpoint'):
        load_checkpoint(checkpoint_path)

    assert not marker_path.exists()


def test_explicit_trusted_loader_warns_before_loading_legacy_pickle(tmp_path):
    checkpoint_path = tmp_path / 'trusted-legacy.pth'
    marker_path = tmp_path / 'executed.txt'
    _save_malicious_checkpoint(checkpoint_path, marker_path)

    with pytest.warns(TrustedCheckpointWarning, match='execute Python code'):
        checkpoint = load_checkpoint(checkpoint_path, trusted=True)

    assert marker_path.read_text(encoding='utf-8') == 'executed'
    assert torch.equal(checkpoint['state_dict']['weight'], torch.ones(1))


def test_nested_policy_restores_outer_environment_and_mmengine_loader(monkeypatch):
    safe_env = 'TORCH_FORCE_WEIGHTS_ONLY_LOAD'
    unsafe_env = 'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'
    monkeypatch.setenv(safe_env, 'external-safe')
    monkeypatch.setenv(unsafe_env, 'external-unsafe')
    original_torch_load = torch.load
    original_local_loader = CheckpointLoader._schemes['']

    with checkpoint_loading_policy():
        outer_local_loader = CheckpointLoader._schemes['']
        assert outer_local_loader is not original_local_loader
        assert torch.load is original_torch_load
        assert os.environ[safe_env] == '1'
        assert unsafe_env not in os.environ

        with pytest.warns(TrustedCheckpointWarning, match='execute Python code'):
            with checkpoint_loading_policy(trusted=True):
                assert CheckpointLoader._schemes[''] is not outer_local_loader
                assert torch.load is original_torch_load
                assert safe_env not in os.environ
                assert os.environ[unsafe_env] == '1'

        assert CheckpointLoader._schemes[''] is outer_local_loader
        assert torch.load is original_torch_load
        assert os.environ[safe_env] == '1'
        assert unsafe_env not in os.environ

    assert CheckpointLoader._schemes[''] is original_local_loader
    assert torch.load is original_torch_load
    assert os.environ[safe_env] == 'external-safe'
    assert os.environ[unsafe_env] == 'external-unsafe'


def test_policy_restores_environment_and_loader_after_exception(monkeypatch):
    safe_env = 'TORCH_FORCE_WEIGHTS_ONLY_LOAD'
    unsafe_env = 'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'
    monkeypatch.setenv(safe_env, 'external-safe')
    monkeypatch.delenv(unsafe_env, raising=False)
    original_torch_load = torch.load
    original_local_loader = CheckpointLoader._schemes['']

    with pytest.raises(RuntimeError, match='sentinel'):
        with checkpoint_loading_policy():
            assert torch.load is original_torch_load
            raise RuntimeError('sentinel')

    assert CheckpointLoader._schemes[''] is original_local_loader
    assert torch.load is original_torch_load
    assert os.environ[safe_env] == 'external-safe'
    assert unsafe_env not in os.environ


def test_trusted_warning_as_error_restores_every_policy_surface(monkeypatch):
    """A warning filter must not turn policy entry into leaked global state."""
    safe_env = 'TORCH_FORCE_WEIGHTS_ONLY_LOAD'
    unsafe_env = 'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'
    monkeypatch.setenv(safe_env, 'external-safe')
    monkeypatch.setenv(unsafe_env, 'external-unsafe')
    original_torch_load = torch.load
    original_local_loader = CheckpointLoader._schemes['']

    with warnings.catch_warnings():
        warnings.simplefilter('error', TrustedCheckpointWarning)
        with pytest.raises(TrustedCheckpointWarning, match='execute Python code'):
            with checkpoint_loading_policy(trusted=True):
                pytest.fail('warning-as-error must abort policy entry')

    assert not trusted_checkpoint_loading_enabled()
    assert CheckpointLoader._schemes[''] is original_local_loader
    assert torch.load is original_torch_load
    assert os.environ[safe_env] == 'external-safe'
    assert os.environ[unsafe_env] == 'external-unsafe'


def test_safe_and_trusted_threads_are_serialized_without_global_leaks(monkeypatch):
    safe_env = 'TORCH_FORCE_WEIGHTS_ONLY_LOAD'
    unsafe_env = 'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'
    monkeypatch.setenv(safe_env, 'external-safe')
    monkeypatch.setenv(unsafe_env, 'external-unsafe')
    original_torch_load = torch.load
    original_local_loader = CheckpointLoader._schemes['']
    safe_entered = threading.Event()
    release_safe = threading.Event()
    trusted_attempting = threading.Event()
    trusted_entered = threading.Event()
    observations = []
    errors = []

    def safe_worker():
        try:
            with checkpoint_loading_policy():
                observations.append((
                    'safe',
                    trusted_checkpoint_loading_enabled(),
                    os.environ.get(safe_env),
                    os.environ.get(unsafe_env),
                    CheckpointLoader._schemes[''] is original_local_loader,
                    torch.load is original_torch_load,
                ))
                safe_entered.set()
                if not release_safe.wait(timeout=2):
                    raise AssertionError('safe worker timed out waiting for release')
        except BaseException as exc:  # surface thread failures in the test
            errors.append(exc)
            safe_entered.set()
            release_safe.set()

    def trusted_worker():
        try:
            if not safe_entered.wait(timeout=2):
                raise AssertionError('trusted worker did not observe safe entry')
            trusted_attempting.set()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', TrustedCheckpointWarning)
                with checkpoint_loading_policy(trusted=True):
                    observations.append((
                        'trusted',
                        trusted_checkpoint_loading_enabled(),
                        os.environ.get(safe_env),
                        os.environ.get(unsafe_env),
                        CheckpointLoader._schemes[''] is original_local_loader,
                        torch.load is original_torch_load,
                    ))
                    trusted_entered.set()
        except BaseException as exc:  # surface thread failures in the test
            errors.append(exc)
            trusted_entered.set()

    safe_thread = threading.Thread(target=safe_worker, name='checkpoint-safe')
    trusted_thread = threading.Thread(
        target=trusted_worker, name='checkpoint-trusted')
    safe_thread.start()
    assert safe_entered.wait(timeout=2)
    trusted_thread.start()
    assert trusted_attempting.wait(timeout=2)
    assert not trusted_entered.wait(timeout=0.05)
    release_safe.set()
    safe_thread.join(timeout=2)
    trusted_thread.join(timeout=2)

    assert not safe_thread.is_alive()
    assert not trusted_thread.is_alive()
    assert errors == []
    assert observations == [
        ('safe', False, '1', None, False, True),
        ('trusted', True, None, '1', False, True),
    ]
    assert not trusted_checkpoint_loading_enabled()
    assert CheckpointLoader._schemes[''] is original_local_loader
    assert torch.load is original_torch_load
    assert os.environ[safe_env] == 'external-safe'
    assert os.environ[unsafe_env] == 'external-unsafe'


def test_real_mmengine_message_hub_checkpoint_loads_safely(tmp_path):
    checkpoint_path = tmp_path / 'mmengine-message-hub.pth'
    source_hub = MessageHub.get_instance(
        f'checkpoint-source-{tmp_path.name}')
    source_hub.update_scalar('loss', 1.0, resumed=True)
    source_hub.update_scalar('loss', 3.0, resumed=True)
    source_hub.update_info('epoch', 4, resumed=True)
    torch.save({
        'state_dict': {'weight': torch.ones(2)},
        'message_hub': source_hub.state_dict(),
    }, checkpoint_path)

    checkpoint = load_checkpoint(checkpoint_path)
    loaded_history = checkpoint['message_hub']['log_scalars']['loss']
    assert isinstance(loaded_history, HistoryBuffer)
    assert loaded_history.mean() == pytest.approx(2.0)

    restored_hub = MessageHub.get_instance(
        f'checkpoint-restored-{tmp_path.name}')
    restored_hub.load_state_dict(checkpoint['message_hub'])
    assert restored_hub.get_scalar('loss').mean() == pytest.approx(2.0)
    assert restored_hub.get_info('epoch') == 4


def test_pre26_runtime_rejects_mmengine_metadata_without_tuple_alias(
    monkeypatch,
    tmp_path,
):
    checkpoint_path = tmp_path / 'mmengine-pre26.pth'
    source_hub = MessageHub.get_instance(
        f'checkpoint-pre26-{tmp_path.name}')
    source_hub.update_scalar('loss', 1.0, resumed=True)
    torch.save({
        'state_dict': {'weight': torch.ones(1)},
        'message_hub': source_hub.state_dict(),
    }, checkpoint_path)
    monkeypatch.setattr(torch, '__version__', '2.5.1')

    with pytest.raises(UnsafeCheckpointError, match='--trusted-checkpoint'):
        load_checkpoint(checkpoint_path)


def test_mmengine_local_loader_obeys_policy_without_replacing_torch_load(tmp_path):
    checkpoint_path = tmp_path / 'mmengine-local.pth'
    marker_path = tmp_path / 'executed.txt'
    _save_malicious_checkpoint(checkpoint_path, marker_path)
    original_torch_load = torch.load
    original_local_loader = CheckpointLoader._schemes['']

    with checkpoint_loading_policy():
        assert torch.load is original_torch_load
        assert CheckpointLoader._schemes[''] is not original_local_loader
        with pytest.raises(UnsafeCheckpointError):
            CheckpointLoader.load_checkpoint(
                str(checkpoint_path), map_location='cpu', logger=None)

    assert torch.load is original_torch_load
    assert CheckpointLoader._schemes[''] is original_local_loader
    assert not marker_path.exists()

    with pytest.warns(TrustedCheckpointWarning, match='execute Python code'):
        with checkpoint_loading_policy(trusted=True):
            checkpoint = CheckpointLoader.load_checkpoint(
                str(checkpoint_path), map_location='cpu', logger=None)

    assert torch.load is original_torch_load
    assert marker_path.read_text(encoding='utf-8') == 'executed'
    assert torch.equal(checkpoint['state_dict']['weight'], torch.ones(1))

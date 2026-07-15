"""Regression tests for scoped checkpoint policy in public entry points."""

from __future__ import annotations

import ast
import importlib.util
import warnings
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import baoiad
import baoiad.checkpoint
import baoiad.config
import baoiad.runtime
from baoiad.checkpoint import TrustedCheckpointWarning


ROOT = Path(__file__).resolve().parents[1]
ENTRY_POINTS = [
    ROOT / 'tools' / 'train.py',
    ROOT / 'tools' / 'test.py',
    ROOT / 'tools' / 'benchmark.py',
    ROOT / 'tools' / 'benchmark_speed.py',
    ROOT / 'tools' / 'train_ast.py',
    ROOT / 'tools' / 'train_regad_strict.py',
    ROOT / 'tools' / 'train_vitad_exact_order.py',
    ROOT / 'baoiad' / 'utils' / 'alignment_probe.py',
]


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f'_checkpoint_test_{path.stem}', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _attribute_name(node: ast.AST) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return '.'.join(reversed(parts))


def test_entry_points_do_not_assign_global_torch_or_mps_functions():
    forbidden = {
        'torch.load',
        'torch.backends.mps.is_available',
        'torch.backends.mps.is_built',
    }
    violations = []
    for path in ENTRY_POINTS:
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        for node in ast.walk(tree):
            targets = node.targets if isinstance(node, ast.Assign) else []
            if isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                targets = [node.target]
            for target in targets:
                name = _attribute_name(target)
                if name in forbidden:
                    violations.append(f'{path.relative_to(ROOT)}:{node.lineno}: {name}')
    assert violations == []


def test_checkpoint_aware_entry_points_expose_explicit_trust_flag():
    for path in ENTRY_POINTS:
        source = path.read_text(encoding='utf-8')
        assert '--trusted-checkpoint' in source, path.relative_to(ROOT)


def test_importing_entry_points_preserves_torch_load_identity():
    original = torch.load
    import_only = [
        ROOT / 'tools' / 'train.py',
        ROOT / 'tools' / 'test.py',
        ROOT / 'tools' / 'train_regad_strict.py',
        ROOT / 'tools' / 'train_vitad_exact_order.py',
        ROOT / 'baoiad' / 'utils' / 'alignment_probe.py',
    ]
    for path in import_only:
        _load_module(path)
        assert torch.load is original, path.relative_to(ROOT)


def test_specialized_trainers_use_baoiad_registry_without_iadbench():
    specialized_trainers = [
        ROOT / 'tools' / 'train_regad_strict.py',
        ROOT / 'tools' / 'train_vitad_exact_order.py',
    ]
    for path in specialized_trainers:
        source = path.read_text(encoding='utf-8')
        tree = ast.parse(source, filename=str(path))
        imported_roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(
                    alias.name.partition('.')[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.partition('.')[0])

        assert 'iadbench' not in imported_roots, path.relative_to(ROOT)
        assert 'iadbench' not in source, path.relative_to(ROOT)
        assert 'from baoiad.registry import' in source, path.relative_to(ROOT)
        assert 'init_default_scope' in source, path.relative_to(ROOT)


def test_ast_forwards_trusted_checkpoint_to_both_stages(monkeypatch, tmp_path):
    module = _load_module(ROOT / 'tools' / 'train_ast.py')
    captured = []

    def fake_run(cmd, **kwargs):
        captured.append((cmd, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, 'run', fake_run)
    args = SimpleNamespace(
        cpu=False,
        offline=False,
        trusted_checkpoint=True,
        resume=False,
        cfg_options=None,
    )
    module._run_stage('teacher', 'config.py', str(tmp_path), args, [])
    module._run_stage('student', 'config.py', str(tmp_path), args, [])

    assert len(captured) == 2
    assert all('--trusted-checkpoint' in cmd for cmd, _ in captured)


def test_benchmark_forwards_runtime_policy_flags():
    module = _load_module(ROOT / 'tools' / 'benchmark.py')
    cmd = ['python', 'tools/train.py', 'config.py']

    module._add_runtime_cli_flags(
        cmd,
        offline=True,
        trusted_checkpoint=True,
    )

    assert cmd == [
        'python',
        'tools/train.py',
        '--offline',
        '--trusted-checkpoint',
        'config.py',
    ]


def _observe_real_checkpoint_policy(monkeypatch, events, active):
    real_policy = baoiad.checkpoint.checkpoint_loading_policy

    @contextmanager
    def observed_policy(trusted):
        events.append(('policy-enter', trusted))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', TrustedCheckpointWarning)
            with real_policy(trusted):
                active['value'] = True
                try:
                    yield
                finally:
                    active['value'] = False
        events.append(('policy-exit', trusted))

    monkeypatch.setattr(
        baoiad.checkpoint, 'checkpoint_loading_policy', observed_policy)


@pytest.mark.parametrize('trusted', [False, True])
def test_train_resume_runs_inside_requested_checkpoint_policy(
    monkeypatch,
    tmp_path,
    trusted,
):
    module = _load_module(ROOT / 'tools' / 'train.py')
    checkpoint_path = tmp_path / 'latest.pth'
    (tmp_path / 'last_checkpoint').write_text(
        str(checkpoint_path), encoding='utf-8')
    cfg = module.Config(dict(work_dir=str(tmp_path)))
    args = SimpleNamespace(
        config='config.py',
        work_dir=None,
        resume=True,
        cpu=False,
        trusted_checkpoint=trusted,
        offline=False,
        cfg_options=None,
    )
    events = []
    active = {'value': False}

    class FakeRunner:
        def resume(self, path):
            assert active['value']
            events.append(('resume', path))

        def train(self):
            assert active['value']
            events.append(('train', None))

    def fake_from_cfg(received_cfg):
        assert active['value']
        assert received_cfg is cfg
        events.append(('from-cfg', None))
        return FakeRunner()

    monkeypatch.setattr(module, 'parse_args', lambda: args)
    monkeypatch.setattr(
        module.Config, 'fromfile', staticmethod(lambda _: cfg))
    monkeypatch.setattr(
        module.Runner, 'from_cfg', staticmethod(fake_from_cfg))
    monkeypatch.setattr(baoiad, 'register_all_modules', lambda: None)
    monkeypatch.setattr(
        baoiad.runtime, 'configure_offline_mode', lambda _: None)
    monkeypatch.setattr(
        baoiad.config, 'apply_data_root_overrides', lambda _: None)
    _observe_real_checkpoint_policy(monkeypatch, events, active)

    module.main()

    assert events == [
        ('policy-enter', trusted),
        ('from-cfg', None),
        ('resume', str(checkpoint_path)),
        ('train', None),
        ('policy-exit', trusted),
    ]


@pytest.mark.parametrize('trusted', [False, True])
def test_evaluation_runs_inside_requested_checkpoint_policy(
    monkeypatch,
    tmp_path,
    trusted,
):
    module = _load_module(ROOT / 'tools' / 'test.py')
    checkpoint_path = tmp_path / 'evaluation.pth'
    cfg = module.Config(dict(work_dir=str(tmp_path)))
    args = SimpleNamespace(
        config='config.py',
        checkpoint=str(checkpoint_path),
        work_dir=None,
        cpu=False,
        trusted_checkpoint=trusted,
        offline=False,
        cfg_options=None,
    )
    events = []
    active = {'value': False}

    class FakeRunner:
        def test(self):
            assert active['value']
            events.append(('test', None))

    def fake_from_cfg(received_cfg):
        assert active['value']
        assert received_cfg is cfg
        assert received_cfg.load_from == str(checkpoint_path)
        events.append(('from-cfg', None))
        return FakeRunner()

    monkeypatch.setattr(module, 'parse_args', lambda: args)
    monkeypatch.setattr(
        module.Config, 'fromfile', staticmethod(lambda _: cfg))
    monkeypatch.setattr(
        module.Runner, 'from_cfg', staticmethod(fake_from_cfg))
    monkeypatch.setattr(baoiad, 'register_all_modules', lambda: None)
    monkeypatch.setattr(
        baoiad.runtime, 'configure_offline_mode', lambda _: None)
    monkeypatch.setattr(
        baoiad.config, 'apply_data_root_overrides', lambda _: None)
    _observe_real_checkpoint_policy(monkeypatch, events, active)

    module.main()

    assert events == [
        ('policy-enter', trusted),
        ('from-cfg', None),
        ('test', None),
        ('policy-exit', trusted),
    ]

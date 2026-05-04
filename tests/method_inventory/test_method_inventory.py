"""Focused tests for the repo-local method inventory validator."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_inventory_module():
    script_path = ROOT / 'tools' / 'check_method_inventory.py'
    spec = importlib.util.spec_from_file_location('method_inventory_validator', script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_inventory_matches_37_method_contract():
    inventory = _load_inventory_module()
    assert inventory.validate_inventory(ROOT) == []


def test_default_residual_scan_is_clean_for_repo_docs():
    inventory = _load_inventory_module()
    assert inventory.residual_hits(ROOT) == []


def test_residual_scan_flags_public_bundle_references(tmp_path):
    inventory = _load_inventory_module()
    sample = tmp_path / 'sample.md'
    sample.write_text('docs/evidence and paper-facing Track Status closed caveat ../baoiad-paper\n', encoding='utf-8')

    hits = inventory.residual_hits(tmp_path, ('sample.md',))

    labels = {hit.label for hit in hits}
    assert 'docs/evidence' in labels
    assert 'paper-facing' in labels
    assert '../baoiad-paper' in labels


def test_cli_default_path_passes(capsys):
    inventory = _load_inventory_module()

    assert inventory.main([]) == 0

    captured = capsys.readouterr()
    assert 'PASS method inventory validation' in captured.out
    assert 'method entries: 37' in captured.out

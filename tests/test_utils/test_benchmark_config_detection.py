"""Tests for repo-local benchmark method selection."""

import importlib.util
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _method_slugs():
    namespace = {}
    exec((ROOT / 'baoiad' / 'method_inventory.py').read_text(), namespace)
    return tuple(entry.slug for entry in namespace['METHODS'])

METHOD_SLUGS = _method_slugs()

@lru_cache(maxsize=1)
def _load_benchmark_module():
    benchmark_path = ROOT / 'tools' / 'benchmark.py'
    spec = importlib.util.spec_from_file_location('baoiad_benchmark', benchmark_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_benchmark_all_methods_uses_repo_inventory():
    benchmark = _load_benchmark_module()
    assert benchmark.get_all_methods() == list(_method_slugs())
    assert len(benchmark.get_all_methods()) == 37


def test_repo_inventory_configs_exist():
    for slug in METHOD_SLUGS:
        assert (ROOT / 'configs' / slug).is_dir(), slug
        assert list((ROOT / 'configs' / slug).glob('*.py')), slug
    public_dirs = {p.name for p in (ROOT / 'configs').iterdir() if p.is_dir() and not p.name.startswith('_')}
    assert public_dirs == set(_method_slugs())


def test_benchmark_priority_is_limited_to_repo_methods():
    benchmark = _load_benchmark_module()
    assert set(benchmark._METHOD_CONFIG_PRIORITY) == set(_method_slugs())


def test_representative_methods_prefer_repo_configs():
    benchmark = _load_benchmark_module()
    expected = {
        'patchcore': 'patchcore_wrn50_256_mvtec_strict.py',
        'rd': 'rd_wrn50_256_mvtec_strict.py',
        'rdpp': 'rdpp_wrn50_256_mvtec_strict.py',
        'uflow': 'uflow_mcait_448_mvtec_strict.py',
        'vitad': 'vitad_256_mvtec_strict.py',
        'winclip': 'winclip_256_mvtec.py',
    }
    for slug, filename in expected.items():
        assert benchmark.find_config(slug).endswith(filename)

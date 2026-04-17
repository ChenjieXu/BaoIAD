"""Tests for CFlow benchmark metadata and selector behavior."""

import importlib.util
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def _load_benchmark_module():
    benchmark_path = ROOT / 'tools' / 'benchmark.py'
    spec = importlib.util.spec_from_file_location('baoiad_benchmark_cflow', benchmark_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cflow_benchmark_prefers_strict_config():
    benchmark = _load_benchmark_module()
    config_path = benchmark.find_config('cflow')
    assert config_path.endswith('cflow_mvtec_strict.py')


def test_cflow_strict_benchmark_uses_best_per_metric_selector():
    benchmark = _load_benchmark_module()
    config_path = ROOT / 'configs' / 'cflow' / 'cflow_mvtec_strict.py'
    assert benchmark.benchmark_result_selector(str(config_path)) == {
        'mode': 'best_per_metric',
        'metrics': ['image_auroc', 'pixel_auroc', 'aupro'],
    }


def test_cflow_strict_benchmark_preserves_workers():
    benchmark = _load_benchmark_module()
    config_path = ROOT / 'configs' / 'cflow' / 'cflow_mvtec_strict.py'
    assert benchmark.keep_dataloader_workers(str(config_path)) is True


def test_parse_metrics_can_select_best_per_metric():
    benchmark = _load_benchmark_module()
    output = '\n'.join([
        'Epoch(val) [1][5/5] ad/image_auroc: 0.7000 ad/pixel_auroc: 0.6000 ad/aupro: 0.4000',
        'Epoch(val) [2][5/5] ad/image_auroc: 0.6800 ad/pixel_auroc: 0.6500 ad/aupro: 0.3900',
        'Epoch(val) [3][5/5] ad/image_auroc: 0.6900 ad/pixel_auroc: 0.6200 ad/aupro: 0.4300',
    ])
    metrics = benchmark.parse_metrics(
        output,
        selector=dict(mode='best_per_metric', metrics=['image_auroc', 'pixel_auroc', 'aupro']),
    )

    assert metrics['image_auroc'] == 0.70
    assert metrics['pixel_auroc'] == 0.65
    assert metrics['aupro'] == 0.43

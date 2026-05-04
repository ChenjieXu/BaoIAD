"""Targeted benchmark-config tests for GANomaly strict alignment."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_benchmark_module():
    import importlib.util

    module_path = ROOT / 'tools' / 'benchmark.py'
    spec = importlib.util.spec_from_file_location('baoiad_test_benchmark_ganomaly', module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_ganomaly_benchmark_prefers_strict_config():
    benchmark = _load_benchmark_module()
    config_path = benchmark.find_config('ganomaly')
    assert config_path.endswith('ganomaly_256_mvtec_strict.py')


def test_ganomaly_strict_config_uses_image_only_metrics():
    benchmark = _load_benchmark_module()
    config_path = ROOT / 'configs' / 'ganomaly' / 'ganomaly_256_mvtec_strict.py'
    cfg = benchmark._load_config(str(config_path))

    assert cfg.benchmark_multi_class is False
    assert cfg.benchmark_keep_dataloader_workers is True
    assert cfg.benchmark_preserve_checkpoint_hooks is True
    assert cfg.benchmark_result_selector == {'mode': 'best', 'metric': 'image_auroc'}
    assert cfg.model['strict'] is True
    assert cfg.train_cfg['max_epochs'] == 15
    assert cfg.train_cfg['val_interval'] == 1
    metric_cfg = cfg.test_evaluator['metrics'][0]
    assert metric_cfg['type'] == 'AnomalyDetectionMetric'
    assert metric_cfg['metrics'] == ['image_auroc', 'image_f1max', 'image_ap', 'image_fpr@95tpr']
    assert metric_cfg['normalize_image_scores'] is True
    assert 'pixel_auroc' not in metric_cfg['metrics']
    assert 'type' not in cfg.test_evaluator
    assert cfg.optim_wrapper['constructor'] == 'GanomalyOptimWrapperConstructor'
    assert cfg.default_hooks['checkpoint']['save_best'] == 'ad/image_auroc'

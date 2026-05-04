#!/usr/bin/env python3
"""BaoIAD: Unified benchmark runner using mmengine Runner + configs.

Usage:
    # Run specific methods on all MVTec AD categories
    python tools/benchmark.py --data_root data/mvtec_ad --categories all --methods patchcore rd

    # Run all methods on GPU (set CUDA_VISIBLE_DEVICES externally for GPU selection)
    CUDA_VISIBLE_DEVICES=2 python tools/benchmark.py --data_root data/mvtec_ad --categories all --methods all

    # Override timeout for heavy methods
    python tools/benchmark.py --data_root data/mvtec_ad --categories all --methods efficientad --timeout 7200
"""
import argparse
from collections.abc import Mapping, Sequence
import copy
import glob
import inspect
import json
import os
import re
import signal
import subprocess
import sys
import time
from functools import lru_cache

from mmengine.config import Config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

ALL_CATEGORIES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper',
]

VISA_CATEGORIES = [
    'candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
    'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4',
    'pipe_fryum',
]

_METHOD_CONFIG_PRIORITY = {
    'aaclip': [
        'aaclip_vitl14_336_518_mvtec_strict.py',
        'aaclip_vitl14_336_256_mvtec.py',
    ],
    # AdaCLIP's canonical strict mainline follows the official auxiliary-data
    # training protocol (VisA + ClinicDB -> MVTec). The checkpoint-only
    # evaluation config is retained as a strict-eval sidecar, not the default.
    'adaclip': [
        'adaclip_vitl14_336_518_mvtec_strict.py',
        'adaclip_vitl14_336_518_visa_clinicdb_train_mvtec.py',
        'adaclip_vitl14_336_518_visa_train_mvtec.py',
        'adaclip_vitl14_336_518_mvtec_train_mvtec.py',
        'adaclip_vitl14_336_256_mvtec.py',
    ],
    # Strict AnomalyCLIP alignment follows the official trainable
    # VisA->MVTec protocol.
    'anomalyclip': [
        'anomalyclip_vitl14_336_518_mvtec_strict.py',
        'anomalyclip_vitl14_336_518_visa_train_mvtec.py',
        'anomalyclip_vitl14_336_256_mvtec.py',
    ],
    'csflow': [
        'csflow_256_mvtec_strict.py',
        'csflow_256_mvtec.py',
    ],
    'anovl': [
        'anovl_vitb16plus_240_mvtec_strict.py',
        'anovl_vitb16plus_240_mvtec.py',
        'anovl_vitb16_224_mvtec.py',
    ],
    # ResAD's strict final mainline now follows the official VisA->MVTec
    # wrapper path. The generic benchmark runner should default to the budget
    # MVTec sidecar config instead.
    'resad': [
        'resad_wrn50_256_mvtec_benchmark.py',
        'resad_official_visa_to_mvtec.py',
        'resad_wrn50_256_mvtec_strict.py',
        'resad_wrn50_256_mvtec.py',
    ],
    'saaplus': [
        'saaplus_400_mvtec_strict.py',
        'saaplus_256_mvtec.py',
    ],
    'dinomaly': [
        'dinomaly_392_mvtec_strict.py',
        'dinomaly_256_mvtec.py',
    ],
    # DFM's repository-default mainline should now follow the explicit strict
    # tv_in1k config; the stronger tv2_in1k path is retained only as a
    # best-repro sidecar archive.
    'dfm': [
        'dfm_256_mvtec_strict.py',
        'dfm_256_mvtec.py',
        'dfm_256_mvtec_racm.py',
    ],
    'graphcore': [
        'graphcore_vig_ti_224_mvtec_strict.py',
        'graphcore_wrn50_256_mvtec.py',
    ],
    'realnet': [
        'realnet_wrn50_256_mvtec_strict.py',
        'realnet_wrn50_256_mvtec_benchmark.py',
    ],
    'memseg': [
        'memseg_rn18_256_mvtec_strict.py',
        'memseg_rn18_256_mvtec.py',
    ],
    # MemAE strict alignment now follows the official video datasets.
    # The generic benchmark runner is still MVTec-category oriented, so this
    # priority only governs the default strict config identity; run-time guard
    # below still refuses generic benchmark.py execution for strict MemAE.
    'memae': [
        'memae_ucsdped2_256_official.py',
        'memae_ucsdped1_256_official.py',
        'memae_avenue_256_official.py',
        'memae_wrn50_256_mvtec.py',
        'memae_wrn50_256_mvtec_adapted.py',
    ],
    'efficientad': [
        'efficientad_256_mvtec_strict.py',
        'efficientad_wrn50_256_mvtec.py',
    ],
    'uninet': [
        'uninet_256_mvtec_strict.py',
        'uninet_256_mvtec.py',
    ],
    'anomalydino': [
        'anomalydino_vitb14_448_mvtec_strict.py',
        'anomalydino_vitb14_448_mvtec.py',
        'anomalydino_vitb14_448_mvtec_no_pca.py',
    ],
    # InvAD strict alignment follows ADer's MUAD protocol, not the historical
    # single-class bottle config.
    'invad': [
        'invad_wrn50_256_mvtec_strict.py',
        'invad_wrn50_256_mvtec.py',
    ],
    # UniAD strict alignment follows the ADer MUAD mainline. The canonical
    # config keeps the historical `wrn50` filename only as a compatibility
    # alias; its actual backbone is EfficientNet-B4.
    'uniad': [
        'uniad_wrn50_256_mvtec_strict.py',
        'uniad_wrn50_256_mvtec.py',
        'uniad_effnet_b4_256_mvtec.py',
    ],
    'spade': [
        'spade_wrn50_256_mvtec_strict.py',
    ],
    'cflow': [
        'cflow_mvtec_strict.py',
        'cflow_wrn50_256_mvtec.py',
    ],
    # GLASS strict official alignment uses the 288px asset-aware config as
    # the benchmark mainline; the 256px variant is retained only as a fast path.
    'glass': [
        'glass_wrn50_288_mvtec_strict.py',
        'glass_wrn50_256_mvtec.py',
    ],
    'simplenet': [
        'simplenet_wrn50_288_mvtec_strict.py',
        'simplenet_wrn50_256_mvtec.py',
    ],
    'destseg': [
        'destseg_rn18_256_mvtec_strict.py',
        'destseg_wrn50_256_mvtec.py',
    ],
    'musc': [
        'musc_vitl14_336_518_mvtec_strict.py',
        'musc_vitl14_336_518_mvtec_simple.py',
        'musc_dinov2_vitb14_256_mvtec.py',
    ],
    # CutPaste strict alignment follows the official ResNet-18 training path.
    'cutpaste': [
        'cutpaste_rn18_256_mvtec_strict.py',
        'cutpaste_rn18_256_mvtec.py',
        'cutpaste_effnet_b4_256_mvtec.py',
        'cutpaste_wrn50_256_mvtec.py',
    ],
    'stfpm': [
        'stfpm_rn18_256_mvtec_strict.py',
        'stfpm_rn18_256_mvtec.py',
        'stfpm_wrn50_256_mvtec.py',
    ],
    'pyramidflow': [
        'pyramidflow_resnet18_1024_mvtec_strict.py',
        'pyramidflow_fnf_256_mvtec_strict.py',
        'pyramidflow_resnet18_256_mvtec.py',
    ],
    # SuperSimpleNet strict alignment should follow the anomalib-matched
    # optimizer param-group config; the old simplified config is retained only
    # as a historical benchmark baseline.
    'supersimplenet': [
        'supersimplenet_256_mvtec_strict.py',
        'supersimplenet_256_mvtec.py',
    ],
    'ast': [
        'ast_effnet_b5_768_mvtec_strict.py',
        'ast_effnet_b5_768_mvtec.py',
        'ast_effnet_b5_mvtec.py',
        'ast_wrn50_256_mvtec.py',
    ],
    # RD strict official alignment should follow the refrozen RD4AD config.
    'rd': [
        'rd_wrn50_256_mvtec_strict.py',
        'rd_wrn50_256_mvtec.py',
        'rd_wrn50_256_mvtec_unified.py',
    ],
    'rdpp': [
        'rdpp_wrn50_256_mvtec_strict.py',
        'rdpp_wrn50_256_mvtec.py',
    ],
    'uflow': [
        'uflow_mcait_448_mvtec_strict.py',
        'uflow_mcait_448_mvtec.py',
        'uflow_256_mvtec.py',
    ],
    'differnet': [
        'differnet_alexnet_256_mvtec_strict.py',
        'differnet_alexnet_256_mvtec.py',
        'differnet_alexnet_448_mvtec.py',
    ],
    'mambaad': [
        'mambaad_effnet_b4_256_mvtec_strict.py',
        'mambaad_effnet_b4_256_mvtec.py',
    ],
    'univad': [
        'univad_mvtec_strict.py',
        'univad_mvtec.py',
    ],
    'pni': [
        'pni_wrn101_480_mvtec_strict.py',
        'pni_wrn101_480_mvtec.py',
    ],
    'regad': [
        'regad_wrn50_256_mvtec_strict.py',
    ],
}

_CLOSED_STRICT_BENCHMARKS = {
    'memae': dict(
        strict_configs={
            'memae_ucsdped2_256_official.py',
            'memae_ucsdped1_256_official.py',
            'memae_avenue_256_official.py',
            'memae_wrn50_256_mvtec.py',
        },
        reason=(
            'MemAE strict alignment now follows official video datasets. '
            'benchmark.py is MVTec-category oriented and must not be used for '
            'strict MemAE. Use tools/memae_official_video_smoke.sh or direct '
            'train/test with the official video configs instead.'
        ),
    ),
}


def find_config(method):
    """Find the primary config file for a given method name.

    Prefers configs with '256' and 'mvtec' in the name, excludes 'unified'
    and '.bak' variants (matching smoke_test_gpu.sh behavior).
    """
    config_dir = os.path.join(ROOT, 'configs', method)
    if not os.path.isdir(config_dir):
        return None
    all_configs = sorted(glob.glob(os.path.join(config_dir, '*.py')))
    # Filter out unified and bak variants
    candidates = [c for c in all_configs
                  if 'unified' not in os.path.basename(c)
                  and not c.endswith('.bak')]

    preferred_names = _METHOD_CONFIG_PRIORITY.get(method.lower())
    if preferred_names:
        by_name = {os.path.basename(c): c for c in candidates}
        for name in preferred_names:
            if name in by_name:
                return by_name[name]

    # Prefer configs with '256' and 'mvtec' in name
    preferred = [c for c in candidates
                 if '256' in os.path.basename(c)
                 and 'mvtec' in os.path.basename(c)]
    if preferred:
        return preferred[0]
    return candidates[0] if candidates else None


def closed_strict_benchmark_reason(method, config_path):
    """Return a skip reason when a method's strict benchmark path is closed."""
    if not method or not config_path:
        return None
    policy = _CLOSED_STRICT_BENCHMARKS.get(str(method).lower())
    if not policy:
        return None
    config_name = os.path.basename(config_path)
    if config_name in policy.get('strict_configs', set()):
        return policy.get('reason', 'strict benchmark is closed')
    return None


def get_all_methods():
    """List all methods that have configs."""
    config_root = os.path.join(ROOT, 'configs')
    methods = []
    for d in sorted(os.listdir(config_root)):
        if d.startswith('_'):
            continue
        if os.path.isdir(os.path.join(config_root, d)):
            methods.append(d)
    return methods


def is_iter_based(config_path):
    """Check if a config uses iteration-based training (by_epoch=False)."""
    cfg = _load_config(config_path)
    if cfg is not None:
        train_cfg = cfg.get('train_cfg', None)
        if train_cfg is not None:
            if 'max_iters' in train_cfg:
                return True
            by_epoch = train_cfg.get('by_epoch', None)
            if by_epoch is not None:
                return not bool(by_epoch)
    try:
        with open(config_path) as f:
            content = f.read()
        return (
            'train_cfg = dict(by_epoch=False' in content
            or 'train_cfg=dict(by_epoch=False' in content
            or 'train_cfg = dict(type=\'IterBasedTrainLoop\'' in content
            or 'train_cfg = dict(type="IterBasedTrainLoop"' in content
        )
    except Exception:
        return False


@lru_cache(maxsize=None)
def _load_config(config_path):
    """Load a config once for metadata checks."""
    try:
        return Config.fromfile(config_path)
    except Exception:
        return None


@lru_cache(maxsize=None)
def _raw_config_text(config_path):
    try:
        with open(config_path) as f:
            return f.read()
    except Exception:
        return ''


def _search_config_flag(config_path, key):
    """Extract a boolean flag from config, resolving _base_ inheritance.

    Uses Config.fromfile() to resolve the full config hierarchy first,
    then falls back to raw text search for backwards compatibility.
    """
    # Primary: resolve through full config inheritance
    try:
        from mmengine.config import Config
        cfg = Config.fromfile(config_path)
        val = cfg.get(key, None)
        if val is not None:
            return bool(val)
    except Exception:
        pass

    # Fallback: raw text search (does not resolve _base_)
    content = _raw_config_text(config_path)
    if not content:
        return None
    if re.search(rf'{re.escape(key)}\s*=\s*True\b', content):
        return True
    if re.search(rf'{re.escape(key)}\s*=\s*False\b', content):
        return False
    return None


def _search_model_type(config_path):
    """Extract `model.type` from raw config text when possible."""
    content = _raw_config_text(config_path)
    if not content:
        return None
    match = re.search(
        r'model\s*=\s*dict\s*\(\s*type\s*=\s*[\'"]([^\'"]+)[\'"]',
        content,
        flags=re.DOTALL,
    )
    return match.group(1) if match else None


def _search_result_selector(config_path):
    """Extract benchmark_result_selector from raw config text when possible."""
    content = _raw_config_text(config_path)
    if not content:
        return None
    match = re.search(
        r'benchmark_result_selector\s*=\s*dict\s*\((.*?)\)',
        content,
        flags=re.DOTALL,
    )
    if not match:
        return None
    body = match.group(1)
    mode_match = re.search(r'mode\s*=\s*[\'"]([^\'"]+)[\'"]', body)
    metric_match = re.search(r'metric\s*=\s*[\'"]([^\'"]+)[\'"]', body)
    selector = {}
    if mode_match:
        selector['mode'] = mode_match.group(1)
    if metric_match:
        selector['metric'] = metric_match.group(1)
    return selector or None


def _search_train_script(config_path):
    """Extract benchmark_train_script from raw config text when possible."""
    content = _raw_config_text(config_path)
    if not content:
        return None
    match = re.search(
        r'benchmark_train_script\s*=\s*[\'"]([^\'"]+)[\'"]',
        content,
    )
    return match.group(1) if match else None


def _search_string_flag(config_path, key):
    """Extract a simple string config flag from raw config text when possible."""
    content = _raw_config_text(config_path)
    if not content:
        return None
    match = re.search(
        rf'{re.escape(key)}\s*=\s*[\'"]([^\'"]+)[\'"]',
        content,
    )
    return match.group(1) if match else None


def is_multi_class_config(config_path):
    """Check if benchmark should treat this config as a multi-class method."""
    explicit = _search_config_flag(config_path, 'benchmark_multi_class')
    if explicit is not None:
        return explicit

    content = _raw_config_text(config_path)
    if (
        'multi_class=True' in content
        or 'multi_class = True' in content
    ):
        return True

    return False


def keep_train_data_root(config_path):
    """Check if benchmark should preserve the config's train data_root.

    Some official protocols train on auxiliary data (for example VisA) while
    evaluating on MVTec. Those configs must not have their train data_root
    overwritten by ``--data_root``.
    """
    explicit = _search_config_flag(config_path, 'benchmark_keep_train_data_root')
    if explicit is not None:
        return explicit

    return False


def keep_dataloader_workers(config_path):
    """Check if benchmark should preserve dataloader worker settings.

    Most benchmark runs clamp workers to zero for stability and lower memory
    usage. Some CPU-heavy preprocessing pipelines become prohibitively slow
    under that policy and may explicitly opt in to preserving their config
    worker counts.
    """
    explicit = _search_config_flag(config_path, 'benchmark_keep_dataloader_workers')
    if explicit is not None:
        return explicit
    return False


def keep_checkpoint_hooks(config_path):
    """Check if benchmark should preserve checkpoint hook settings.

    Strict alignment configs that depend on best-checkpoint retention should
    opt out of benchmark.py's default checkpoint-disabling policy.
    """
    explicit = _search_config_flag(config_path, 'benchmark_preserve_checkpoint_hooks')
    if explicit is not None:
        return explicit
    return False


def resume_existing_benchmark(config_path):
    """Check if benchmark should pass `--resume` to the train entrypoint."""
    explicit = _search_config_flag(config_path, 'benchmark_resume_existing')
    if explicit is not None:
        return explicit
    return False


def rescale_epoch_schedulers(config_path):
    """Check if benchmark should rescale epoch-based scheduler milestones.

    Some strict configs define scheduler milestones relative to the training
    budget in the official code. When benchmark.py overrides ``max_epochs`` for
    smoke runs, those milestones should be rescaled as well.
    """
    explicit = _search_config_flag(config_path, 'benchmark_rescale_epoch_schedulers')
    if explicit is not None:
        return explicit
    return False


def disable_compile_for_benchmark(config_path):
    """Check if benchmark should explicitly disable torch.compile."""
    explicit = _search_config_flag(config_path, 'benchmark_disable_compile')
    if explicit is not None:
        return explicit
    explicit = _search_config_flag(config_path, 'train_disable_compile')
    if explicit is not None:
        return explicit
    return False


def is_eval_only_config(config_path):
    """Check if benchmark should run this config via `tools/test.py`."""
    explicit = _search_config_flag(config_path, 'benchmark_eval_only')
    if explicit is not None:
        return explicit

    return False


def benchmark_test_after_train(config_path):
    """Check if benchmark should run `tools/test.py` after training."""
    explicit = _search_config_flag(config_path, 'benchmark_test_after_train')
    if explicit is not None:
        return explicit
    return False


def benchmark_checkpoint_source(config_path):
    """Return which checkpoint source benchmark should evaluate."""
    cfg = _load_config(config_path)
    if cfg is not None:
        explicit = cfg.get('benchmark_checkpoint_source', None)
        if explicit is not None:
            return str(explicit)

    explicit = _search_string_flag(config_path, 'benchmark_checkpoint_source')
    if explicit is not None:
        return explicit
    return 'last'


def benchmark_timeout(config_path, default_timeout):
    """Return the effective timeout for one benchmark run.

    Configs may declare a minimum safe timeout via ``benchmark_timeout``.
    The CLI timeout still acts as an override upwards, but benchmark should
    not silently use a smaller timeout than the config explicitly requires.
    """
    cfg = _load_config(config_path)
    configured = None
    if cfg is not None:
        configured = cfg.get('benchmark_timeout', None)
    if configured is None:
        explicit = _search_string_flag(config_path, 'benchmark_timeout')
        configured = int(explicit) if explicit is not None else None

    if configured is None:
        return int(default_timeout)
    return max(int(default_timeout), int(configured))


def benchmark_result_selector(config_path):
    """Return how benchmark should select metrics from train/test output."""
    cfg = _load_config(config_path)
    if cfg is not None:
        explicit = cfg.get('benchmark_result_selector', None)
        if explicit is not None:
            if isinstance(explicit, str):
                return dict(mode=explicit)
            return dict(explicit)

    selector = _search_result_selector(config_path)
    if selector is not None:
        return selector
    return dict(mode='last')


def benchmark_train_script(config_path):
    """Return the train script benchmark should use for this config."""
    cfg = _load_config(config_path)
    if cfg is not None:
        explicit = cfg.get('benchmark_train_script', None)
        if explicit is not None:
            return os.path.join(ROOT, explicit) if not os.path.isabs(explicit) else explicit

    script = _search_train_script(config_path)
    if script is None:
        return os.path.join(ROOT, 'tools', 'train.py')
    return os.path.join(ROOT, script) if not os.path.isabs(script) else script


@lru_cache(maxsize=None)
def configured_benchmark_categories(config_path):
    """Return an optional category subset frozen by the config."""
    cfg = _load_config(config_path)
    if cfg is None:
        return None

    categories = cfg.get('benchmark_categories', None)
    if categories is None:
        return None
    if isinstance(categories, str):
        return [categories]
    return list(categories)


@lru_cache(maxsize=None)
def configured_benchmark_summary_categories(config_path):
    """Return an optional category subset used only for summary statistics."""
    cfg = _load_config(config_path)
    if cfg is None:
        return None

    categories = cfg.get('benchmark_summary_categories', None)
    if categories is None:
        return None
    if isinstance(categories, str):
        return [categories]
    return list(categories)


def benchmark_category_cfg_options(config_path, category):
    """Return extra cfg-options frozen for one benchmark category."""
    cfg = _load_config(config_path)
    if cfg is None:
        return []

    raw_options = cfg.get('benchmark_category_cfg_options', None)
    if raw_options is None:
        return []

    selected = []
    if isinstance(raw_options, Mapping):
        default_options = raw_options.get('__default__', raw_options.get('default', None))
        if default_options is not None:
            selected.extend(_normalize_cfg_option_items(default_options))
        category_options = raw_options.get(category, None)
        if category_options is not None:
            selected.extend(_normalize_cfg_option_items(category_options))
        return selected

    return _normalize_cfg_option_items(raw_options)


def strict_alignment_guard_errors(config_path):
    """Return strict-alignment guard failures for benchmark mainline configs."""
    cfg = _load_config(config_path)
    if cfg is None:
        return []

    config_dir = os.path.dirname(os.path.abspath(config_path))
    graphcore_dir = os.path.join(ROOT, 'configs', 'graphcore')
    model_cfg = cfg.get('model', {})
    if (
        os.path.normpath(config_dir) == os.path.normpath(graphcore_dir)
        and isinstance(model_cfg, Mapping)
        and model_cfg.get('type') == 'GraphCoreDetector'
    ):
        from baoiad.utils.graphcore_alignment import graphcore_strict_alignment_violations

        return graphcore_strict_alignment_violations(model_cfg)
    return []


def _graphcore_train_order_cfg_options(config_path, category):
    cfg = _load_config(config_path)
    if cfg is None:
        return []
    model_cfg = cfg.get('model', {})
    if not isinstance(model_cfg, Mapping) or model_cfg.get('type') != 'GraphCoreDetector':
        return []

    from baoiad.utils.graphcore_alignment import graphcore_explicit_order_cfg_overrides

    overrides = graphcore_explicit_order_cfg_overrides(category)
    return _normalize_cfg_option_items(overrides)


def _normalize_cfg_option_items(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [f'{key}={repr(item)}' for key, item in value.items()]
    if isinstance(value, Sequence):
        normalized = []
        for item in value:
            normalized.extend(_normalize_cfg_option_items(item))
        return normalized
    raise TypeError(f'Unsupported benchmark_category_cfg_options value: {type(value)!r}')


ALL_METRICS = [
    'image_auroc', 'pixel_auroc',
    'image_auroc_mean', 'image_auroc_max',
    'image_f1max', 'pixel_f1max',
    'image_ap', 'pixel_ap',
    'aupro', 'aupimo',
    'image_ece', 'pixel_ece',
    'image_fpr@95tpr',
]


def _prepare_subprocess_env(base_env=None, disable_compile=False):
    """Prepare a subprocess env that always prioritizes this repo root."""
    env_copy = dict(os.environ if base_env is None else base_env)
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_visible:
        env_copy['CUDA_VISIBLE_DEVICES'] = cuda_visible
    existing_pythonpath = env_copy.get('PYTHONPATH', '')
    env_copy['PYTHONPATH'] = (
        ROOT if not existing_pythonpath else f'{ROOT}{os.pathsep}{existing_pythonpath}'
    )
    env_copy.setdefault('OPENBLAS_NUM_THREADS', '1')
    env_copy.setdefault('OMP_NUM_THREADS', '1')
    env_copy.setdefault('MKL_NUM_THREADS', '1')
    env_copy.setdefault('PYTHONWARNINGS', 'ignore')
    if disable_compile:
        env_copy['TORCH_COMPILE_DISABLE'] = '1'
        env_copy['TORCHDYNAMO_DISABLE'] = '1'
    return env_copy

_DIRECT_RUNNER_MODEL_TYPES = {'PatchCore', 'AnomalyDINODetector'}


def _parse_metric_snapshot(line):
    """Parse one metric snapshot from a benchmark log line."""
    metrics = {}
    for key in ALL_METRICS:
        escaped_key = re.escape(key)
        m = re.search(rf'ad/{escaped_key}:\s+([0-9.]+)', line)
        if m:
            metrics[key] = float(m.group(1))
    return metrics


def parse_metrics(output, selector=None):
    """Parse metrics from benchmark output.

    By default returns the last validation snapshot. When the selector mode is
    ``best``, returns the snapshot with the maximum value of the requested
    metric (for example official DifferNet reports the best image AUROC across
    training, not the final epoch).
    """
    selector = selector or {}
    mode = selector.get('mode', 'last')
    metric_name = selector.get('metric', 'image_auroc')
    metric_names = selector.get('metrics', ['image_auroc', 'pixel_auroc'])
    tie_break_metric = selector.get('tie_break_metric', 'image_ap')

    snapshots = []
    for line in output.split('\n'):
        metrics = _parse_metric_snapshot(line)
        if metrics:
            snapshots.append(metrics)

    if not snapshots:
        return {}

    if mode == 'best':
        eligible = [snapshot for snapshot in snapshots if metric_name in snapshot]
        if eligible:
            return max(eligible, key=lambda snapshot: snapshot[metric_name])
    if mode == 'best_per_metric':
        selected = dict(snapshots[-1])
        for per_metric in selector.get('metrics', []):
            eligible = [snapshot for snapshot in snapshots if per_metric in snapshot]
            if eligible:
                selected[per_metric] = max(snapshot[per_metric] for snapshot in eligible)
        return selected
    if mode == 'best_balanced':
        eligible = [
            snapshot for snapshot in snapshots
            if all(metric in snapshot for metric in metric_names)
        ]
        if eligible:
            def _rank(snapshot):
                balanced = sum(snapshot[name] for name in metric_names) / len(metric_names)
                tie_break = snapshot.get(tie_break_metric, float('-inf'))
                return (balanced, *(snapshot[name] for name in metric_names), tie_break)
            return max(eligible, key=_rank)
    return snapshots[-1]


def _should_use_direct_runner(config_path):
    """Use an in-process benchmark path for training-free memory-bank methods."""
    model_type = _search_model_type(config_path)
    if model_type is not None:
        return model_type in _DIRECT_RUNNER_MODEL_TYPES

    cfg = _load_config(config_path)
    if cfg is None:
        return False
    model_cfg = cfg.get('model', {})
    return model_cfg.get('type') in _DIRECT_RUNNER_MODEL_TYPES


def _move_inputs_to_device(inputs, device):
    import torch

    if torch.is_tensor(inputs):
        return inputs.to(device)
    if isinstance(inputs, list):
        return [item.to(device) if torch.is_tensor(item) else item for item in inputs]
    if isinstance(inputs, tuple):
        return tuple(item.to(device) if torch.is_tensor(item) else item for item in inputs)
    return inputs


def _move_data_samples_to_device(data_samples, device):
    moved = []
    for sample in data_samples:
        if hasattr(sample, 'to'):
            moved.append(sample.to(device))
        else:
            moved.append(sample)
    return moved


def _run_direct_patchcore(config_path, data_root, category, device,
                          batch_size, work_dir, extra_cfg_options=None):
    """Benchmark PatchCore in-process without Runner.train()/val loop overhead."""
    os.environ.setdefault('HF_HUB_OFFLINE', '1')

    import torch
    from mmengine.config import Config, DictAction
    from mmengine.registry import init_default_scope
    from mmengine.runner import Runner

    import baoiad  # noqa: F401
    from baoiad.registry import METRICS, MODELS
    from baoiad.utils.alignment_probe import set_global_seed

    cfg = copy.deepcopy(_load_config(config_path) or Config.fromfile(config_path))
    if extra_cfg_options:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--cfg-options', nargs='+', action=DictAction)
        parsed, _ = parser.parse_known_args(['--cfg-options', *extra_cfg_options])
        cfg.merge_from_dict(parsed.cfg_options)

    cfg.work_dir = work_dir
    cfg.train_dataloader.dataset.data_root = data_root
    cfg.test_dataloader.dataset.data_root = data_root
    cfg.val_dataloader.dataset.data_root = data_root
    cfg.train_dataloader.dataset.cls_names = [category]
    cfg.test_dataloader.dataset.cls_names = [category]
    cfg.val_dataloader.dataset.cls_names = [category]
    cfg.train_dataloader.dataset.multi_class = False
    cfg.test_dataloader.dataset.multi_class = False
    cfg.val_dataloader.dataset.multi_class = False
    cfg.train_dataloader.num_workers = 0
    cfg.test_dataloader.num_workers = 0
    cfg.val_dataloader.num_workers = 0
    cfg.train_dataloader.persistent_workers = False
    cfg.test_dataloader.persistent_workers = False
    cfg.val_dataloader.persistent_workers = False
    if batch_size is not None:
        cfg.train_dataloader.batch_size = batch_size
        cfg.test_dataloader.batch_size = batch_size
        cfg.val_dataloader.batch_size = batch_size

    init_default_scope(cfg.get('default_scope', 'baoiad'))
    set_global_seed(cfg.get('randomness', {}).get('seed', 42))

    torch_device = torch.device('cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu')
    model = MODELS.build(cfg.model).to(torch_device)
    train_loader = Runner.build_dataloader(cfg.train_dataloader, seed=42)
    test_loader = Runner.build_dataloader(cfg.test_dataloader, seed=42)
    metric = METRICS.build(cfg.test_evaluator)

    build_memory_bank = getattr(model, 'build_memory_bank')
    if not getattr(model, 'build_memory_bank_from_dataloader_only', False):
        model.train()
        with torch.no_grad():
            for batch in train_loader:
                inputs = _move_inputs_to_device(batch['inputs'], torch_device)
                data_samples = _move_data_samples_to_device(batch.get('data_samples', []), torch_device)
                model(inputs, data_samples, mode='loss')

    if len(inspect.signature(build_memory_bank).parameters) > 0:
        build_memory_bank(train_loader)
    else:
        build_memory_bank()
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            inputs = _move_inputs_to_device(batch['inputs'], torch_device)
            data_samples = _move_data_samples_to_device(batch.get('data_samples', []), torch_device)
            outputs = model(inputs, data_samples, mode='predict')
            metric.process(batch, outputs)

    metrics = metric.compute_metrics(metric.results)
    return metrics, 'direct_patchcore'


def _find_benchmark_checkpoint(work_dir, source='last'):
    """Find the checkpoint benchmark should evaluate after training."""
    if source == 'best':
        best_checkpoints = sorted(glob.glob(os.path.join(work_dir, 'best_*.pth')))
        if best_checkpoints:
            return best_checkpoints[-1]
        raise FileNotFoundError(f'No best checkpoint found under {work_dir}')

    last_checkpoint = os.path.join(work_dir, 'last_checkpoint')
    if os.path.exists(last_checkpoint):
        with open(last_checkpoint, 'r') as f:
            return f.read().strip()

    epoch_checkpoints = sorted(glob.glob(os.path.join(work_dir, 'epoch_*.pth')))
    if epoch_checkpoints:
        return epoch_checkpoints[-1]

    best_checkpoints = sorted(glob.glob(os.path.join(work_dir, 'best_*.pth')))
    if best_checkpoints:
        return best_checkpoints[-1]

    raise FileNotFoundError(f'No checkpoint found under {work_dir}')


def run_method(config_path, data_root, category, device, epochs,
               batch_size, work_dir, timeout, multi_class=False,
               extra_cfg_options=None):
    """Run a method via tools/train.py using Runner."""
    timeout = benchmark_timeout(config_path, timeout)

    if _should_use_direct_runner(config_path):
        return _run_direct_patchcore(
            config_path=config_path,
            data_root=data_root,
            category=category,
            device=device,
            batch_size=batch_size,
            work_dir=work_dir,
            extra_cfg_options=extra_cfg_options,
        )

    # Prefer the current interpreter so benchmark subprocesses run in the same
    # environment that launched this script. The repo-local .venv may exist but
    # still be missing optional dependencies needed to import baoiad.
    python = sys.executable
    if not python or not os.path.exists(python):
        python = os.path.join(ROOT, '.venv', 'bin', 'python')

    train_script = benchmark_train_script(config_path)
    test_script = os.path.join(ROOT, 'tools', 'test.py')

    # Check if this is a RegAD config (uses cross-category training)
    config_name = os.path.basename(config_path)
    is_regad = 'regad' in config_name.lower()
    eval_only = is_eval_only_config(config_path)
    test_after_train = benchmark_test_after_train(config_path)

    # CutPaste configs wrap the train split in RepeatDataset, so cfg overrides
    # must target the inner dataset rather than the wrapper.
    is_repeat_dataset = 'cutpaste' in config_name.lower()

    guard_errors = strict_alignment_guard_errors(config_path)
    if guard_errors:
        return {}, 'strict alignment guard failed: ' + '; '.join(guard_errors)

    # Build cfg-options to override data root and, for single-class methods, category.
    preserve_train_root = keep_train_data_root(config_path)
    cfg_options = []
    if is_repeat_dataset:
        # RepeatDataset wraps the actual dataset, so set data_root on the inner dataset
        if not preserve_train_root and not eval_only:
            cfg_options.append(f'train_dataloader.dataset.dataset.data_root={data_root}')
        cfg_options.extend([
            f'test_dataloader.dataset.data_root={data_root}',
            f'val_dataloader.dataset.data_root={data_root}',
        ])
    else:
        if not preserve_train_root and not eval_only:
            cfg_options.append(f'train_dataloader.dataset.data_root={data_root}')
        cfg_options.extend([
            f'test_dataloader.dataset.data_root={data_root}',
            f'val_dataloader.dataset.data_root={data_root}',
        ])

    if is_regad:
        # RegAD uses cross-category training: target_cls is the category to EXCLUDE
        cfg_options.extend([
            f'train_dataloader.dataset.target_cls={category}',
            f'test_dataloader.dataset.target_cls={category}',
            f'val_dataloader.dataset.target_cls={category}',
            # For cross-category training, model needs to know target class
            # to load support images from that category's training data
            f'model.target_cls={category}',
            f'model.data_root={data_root}',
        ])
    elif not multi_class:
        if is_repeat_dataset:
            # RepeatDataset wraps the actual dataset
            cfg_options.extend([
                f"train_dataloader.dataset.dataset.cls_names=['{category}']",
                'train_dataloader.dataset.dataset.multi_class=False',
                f"test_dataloader.dataset.cls_names=['{category}']",
                'test_dataloader.dataset.multi_class=False',
                f"val_dataloader.dataset.cls_names=['{category}']",
                'val_dataloader.dataset.multi_class=False',
            ])
        else:
            cfg_options.extend([
                f"train_dataloader.dataset.cls_names=['{category}']",
                'train_dataloader.dataset.multi_class=False',
                f"test_dataloader.dataset.cls_names=['{category}']",
                'test_dataloader.dataset.multi_class=False',
                f"val_dataloader.dataset.cls_names=['{category}']",
                'val_dataloader.dataset.multi_class=False',
            ])

    cfg_options.extend(benchmark_category_cfg_options(config_path, category))
    cfg_options.extend(_graphcore_train_order_cfg_options(config_path, category))

    # Handle epoch override: skip for iter-based methods
    if epochs is not None and not is_iter_based(config_path) and not eval_only:
        cfg_options.append(f'train_cfg.max_epochs={epochs}')
        cfg_options.append(f'train_cfg.val_interval={epochs}')
        cfg_options.append('train_cfg.val_begin=1')
        if rescale_epoch_schedulers(config_path):
            cfg = _load_config(config_path)
            if cfg is not None:
                base_epochs = int(cfg.get('train_cfg', {}).get('max_epochs', epochs))
                schedulers = cfg.get('param_scheduler', [])
                if base_epochs > 0:
                    for idx, scheduler in enumerate(schedulers):
                        if scheduler.get('type') != 'MultiStepLR':
                            continue
                        milestones = scheduler.get('milestones')
                        if not milestones:
                            continue
                        scaled = [
                            max(1, min(epochs, int(epochs * (milestone / base_epochs))))
                            for milestone in milestones
                        ]
                        cfg_options.append(f'param_scheduler.{idx}.milestones={scaled}')

    # NSA-specific: category-specific epochs (hazelnut/metal_nut/screw need 560, others 320)
    # This overrides the default 320 epochs in the config
    if 'nsa' in config_path.lower() and epochs is None and not eval_only:
        if category in ['hazelnut', 'metal_nut', 'screw']:
            cfg_options.append('train_cfg.max_epochs=560')
            cfg_options.append('param_scheduler.0.T_max=560')

    # Only override batch_size if explicitly provided
    if batch_size is not None:
        if not eval_only:
            cfg_options.append(f'train_dataloader.batch_size={batch_size}')
        cfg_options.append(f'test_dataloader.batch_size={batch_size}')

    # Reduce memory usage by default, but allow configs with heavy CPU-side
    # preprocessing to preserve their worker settings explicitly.
    preserve_workers = keep_dataloader_workers(config_path)
    if not preserve_workers:
        if not eval_only:
            cfg_options.append('train_dataloader.num_workers=0')
            cfg_options.append('train_dataloader.persistent_workers=False')
        cfg_options.append('test_dataloader.num_workers=0')
        cfg_options.append('test_dataloader.persistent_workers=False')
        cfg_options.append('val_dataloader.num_workers=0')
        cfg_options.append('val_dataloader.persistent_workers=False')
    preserve_checkpoint = keep_checkpoint_hooks(config_path) or test_after_train
    if not preserve_checkpoint:
        cfg_options.append('default_hooks.checkpoint.interval=1000000')
        cfg_options.append('default_hooks.checkpoint.save_last=False')

    if extra_cfg_options:
        cfg_options.extend(extra_cfg_options)

    disable_compile = disable_compile_for_benchmark(config_path)
    if disable_compile:
        cfg_options.append('runtime_disable_compile=True')

    if eval_only:
        cmd = [python, test_script, config_path,
               '--work-dir', work_dir,
               '--cfg-options'] + cfg_options
    else:
        cmd = [python, train_script, config_path,
               '--work-dir', work_dir,
               '--cfg-options'] + cfg_options
        if resume_existing_benchmark(config_path):
            cmd.insert(3, '--resume')

    if device == 'cpu':
        cmd.insert(2, '--cpu')

    os.umask(0)  # NFS shared env: ensure work dirs are world-writable

    env_copy = _prepare_subprocess_env(disable_compile=disable_compile)
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=ROOT,
            env=env_copy,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                stdout, stderr = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate()
            return {}, f"timeout ({timeout}s)"
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.communicate(timeout=10)
            except Exception:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except Exception:
                    pass
            raise

        output = stdout + stderr

        if process.returncode != 0:
            metrics = parse_metrics(output, selector=benchmark_result_selector(config_path))
            if metrics:
                return metrics, output[-200:]
            return {}, f"exit code {process.returncode}\n{output[-500:]}"

        if test_after_train and not eval_only:
            checkpoint_path = _find_benchmark_checkpoint(
                work_dir,
                source=benchmark_checkpoint_source(config_path),
            )
            test_cmd = [python, test_script, config_path, checkpoint_path,
                        '--work-dir', work_dir,
                        '--cfg-options'] + cfg_options
            if device == 'cpu':
                test_cmd.insert(2, '--cpu')

            test_process = subprocess.Popen(
                test_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=ROOT,
                env=env_copy,
                start_new_session=True,
            )
            try:
                test_stdout, test_stderr = test_process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(test_process.pid, signal.SIGTERM)
                try:
                    test_stdout, test_stderr = test_process.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(test_process.pid, signal.SIGKILL)
                    test_stdout, test_stderr = test_process.communicate()
                return {}, f"test timeout ({timeout}s)"
            except BaseException:
                os.killpg(test_process.pid, signal.SIGTERM)
                try:
                    test_process.communicate(timeout=10)
                except Exception:
                    try:
                        os.killpg(test_process.pid, signal.SIGKILL)
                    except Exception:
                        pass
                raise

            test_output = test_stdout + test_stderr
            test_metrics = parse_metrics(test_output, selector=benchmark_result_selector(config_path))
            if test_process.returncode != 0 and not test_metrics:
                return {}, f"test exit code {test_process.returncode}\n{test_output[-500:]}"
            return test_metrics, test_output[-200:]

        metrics = parse_metrics(output, selector=benchmark_result_selector(config_path))
        return metrics, output[-200:]
    except Exception as e:
        return {}, str(e)


def main():
    parser = argparse.ArgumentParser(description='BaoIAD Benchmark Runner')
    parser.add_argument('--data_root', required=True, help='Dataset root directory')
    parser.add_argument('--categories', nargs='+', default=['bottle'],
                        help='Categories to test (or "all")')
    parser.add_argument('--methods', nargs='+', default=None,
                        help='Methods to test (or "all"). If not set with --config, '
                        'derived from config path.')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override max_epochs (ignored for iter-based methods)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch_size (default: use config value)')
    parser.add_argument('--timeout', type=int, default=3600,
                        help='Timeout per run in seconds (default: 3600)')
    parser.add_argument('--config', default=None,
                        help='Override config path (use with single method)')
    parser.add_argument('--output', default='results/benchmark.json',
                        help='Output JSON file')
    parser.add_argument(
        '--work-dir-root',
        default=os.path.join(ROOT, 'runs', 'benchmark'),
        help='Root directory under which per-method/per-category work dirs are created.',
    )
    parser.add_argument('--cfg-options', nargs='+', default=None,
                        help='Extra cfg-options forwarded to train.py')
    args = parser.parse_args()

    requested_all_categories = 'all' in args.categories
    if requested_all_categories:
        # Auto-detect dataset: if any VISA-specific category has train/ dir,
        # use VISA categories; otherwise default to MVTec AD.
        visa_present = any(
            os.path.isdir(os.path.join(args.data_root, c, 'train'))
            for c in VISA_CATEGORIES
        )
        categories = VISA_CATEGORIES if visa_present else ALL_CATEGORIES
    else:
        categories = args.categories

    # Resolve methods
    if args.methods is None:
        if args.config:
            # Derive method name from config path: configs/<method>/... -> <method>
            methods = [os.path.basename(os.path.dirname(args.config))]
        else:
            methods = ['patchcore', 'rd']
    elif 'all' in args.methods:
        methods = get_all_methods()
    else:
        methods = args.methods

    # Check available categories
    avail_cats = [c for c in categories
                  if os.path.isdir(os.path.join(args.data_root, c, 'train'))]
    if not avail_cats:
        print(f"No categories found in {args.data_root}")
        return

    # Check available methods (must have config)
    method_configs = {}
    for m in methods:
        if args.config:
            cfg = args.config
        else:
            cfg = find_config(m)
        if cfg:
            closed_reason = closed_strict_benchmark_reason(m, cfg)
            if closed_reason:
                print(f"[WARN] Skipping method '{m}': {closed_reason}")
                continue
            method_configs[m] = cfg
        else:
            print(f"[WARN] No config found for method '{m}', skipping")

    if not method_configs:
        print("No valid methods to run")
        return

    print(f"{'=' * 80}")
    print("BaoIAD Benchmark")
    print(f"{'=' * 80}")
    print(f"Categories: {avail_cats}")
    print(f"Methods:    {list(method_configs.keys())}")
    print(f"Device:     {args.device}")
    print(f"Timeout:    {args.timeout}s per run")
    print(f"{'=' * 80}\n")

    results = {}

    def save_results():
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.',
                    exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)

    for method, config_path in method_configs.items():
        results[method] = {}
        is_multi_class = is_multi_class_config(config_path)
        if is_multi_class:
            method_categories = ['all']
        else:
            method_categories = list(avail_cats)
            if requested_all_categories:
                config_categories = configured_benchmark_categories(config_path)
                if config_categories:
                    allowed = set(config_categories)
                    method_categories = [
                        category for category in config_categories
                        if category in allowed and category in avail_cats
                    ]
        summary_categories = None
        if not is_multi_class:
            configured_summary_categories = configured_benchmark_summary_categories(config_path)
            if configured_summary_categories:
                allowed = set(configured_summary_categories)
                summary_categories = [
                    category for category in method_categories
                    if category in allowed
                ]
        iter_based = is_iter_based(config_path)
        mode_text = 'multi-class' if is_multi_class else 'single-class'
        print(f"\n--- {method} ({mode_text}) {'(iter-based)' if iter_based else ''} ---")

        for cat in method_categories:
            work_dir = os.path.join(args.work_dir_root, method, cat)
            print(f"  [{method}] {cat}...", end=' ', flush=True)

            t0 = time.time()
            metrics, info = run_method(
                config_path, args.data_root, cat, args.device,
                args.epochs, args.batch_size, work_dir, args.timeout, is_multi_class,
                args.cfg_options,
            )
            elapsed = time.time() - t0

            if metrics:
                cat_result = dict(metrics)  # store all parsed metrics
                cat_result['time'] = round(elapsed, 1)
                results[method][cat] = cat_result
                ia = metrics.get('image_auroc', 0)
                pa = metrics.get('pixel_auroc', 0)
                f1 = metrics.get('image_f1max', 0)
                print(f"img={ia:.4f} pxl={pa:.4f} f1={f1:.4f} ({elapsed:.0f}s)")
            else:
                results[method][cat] = {'image_auroc': None, 'error': info[:500]}
                print(f"FAILED ({info[:120]})")
            save_results()

            # Free GPU memory between categories to prevent OOM accumulation
            import gc
            gc.collect()
            try:
                import torch as _torch
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
            except ImportError:
                pass

        def _compute_average(selected_categories):
            avg = {}
            num = 0
            selected_results = [
                results[method][category]
                for category in selected_categories
                if category in results[method]
            ]
            for key in ALL_METRICS:
                vals = [
                    result.get(key) for result in selected_results
                    if result.get(key) is not None and not isinstance(result.get(key), str)
                ]
                if vals:
                    avg[key] = round(sum(vals) / len(vals), 4)
                    if key == 'image_auroc':
                        num = len(vals)
            if num > 0:
                avg['num_categories'] = num
            return avg

        avg_metrics = _compute_average(summary_categories or method_categories)
        num_cats = avg_metrics.get('num_categories', 0)
        if num_cats > 0:
            if summary_categories:
                avg_metrics['summary_categories'] = list(summary_categories)
            results[method]['_average'] = avg_metrics
            if summary_categories and list(summary_categories) != list(method_categories):
                all_avg_metrics = _compute_average(method_categories)
                if all_avg_metrics.get('num_categories', 0) > 0:
                    all_avg_metrics['summary_categories'] = list(method_categories)
                    results[method]['_average_all'] = all_avg_metrics
            ia = avg_metrics.get('image_auroc', 0)
            pa = avg_metrics.get('pixel_auroc', 0)
            f1 = avg_metrics.get('image_f1max', 0)
            ap = avg_metrics.get('aupro', 0)
            total_cats = len(method_categories)
            if summary_categories and list(summary_categories) != list(method_categories):
                print(f"  >> {method} official average: img={ia:.4f} pxl={pa:.4f} "
                      f"f1={f1:.4f} aupro={ap:.4f} ({num_cats}/{total_cats} runs, summary subset)")
                all_avg = results[method].get('_average_all', {})
                all_num = all_avg.get('num_categories', 0)
                if all_num > 0:
                    print(f"  >> {method} all-run average archived separately: "
                          f"img={all_avg.get('image_auroc', 0):.4f} "
                          f"pxl={all_avg.get('pixel_auroc', 0):.4f} "
                          f"f1={all_avg.get('image_f1max', 0):.4f} "
                          f"aupro={all_avg.get('aupro', 0):.4f} ({all_num}/{total_cats} runs)")
            else:
                print(f"  >> {method} average: img={ia:.4f} pxl={pa:.4f} "
                      f"f1={f1:.4f} aupro={ap:.4f} ({num_cats}/{total_cats} runs)")
            save_results()

    # Summary table
    print(f"\n{'=' * 110}")
    print(f"{'Method':<15} {'img_AUROC':>10} {'pxl_AUROC':>10} {'img_F1':>10} "
          f"{'img_AP':>10} {'AUPRO':>10} {'FPR@95':>10} {'cats':>5}")
    print('-' * 110)

    for method in method_configs:
        avg = results[method].get('_average', {})
        nc = avg.get('num_categories', 0)
        if nc > 0:
            ia = avg.get('image_auroc', 0)
            pa = avg.get('pixel_auroc', 0)
            f1 = avg.get('image_f1max', 0)
            ap = avg.get('image_ap', 0)
            aupro = avg.get('aupro', 0)
            fpr = avg.get('image_fpr@95tpr', 0)
            print(f"{method:<15} {ia:>10.4f} {pa:>10.4f} {f1:>10.4f} "
                  f"{ap:>10.4f} {aupro:>10.4f} {fpr:>10.4f} {nc:>5}")
        else:
            print(f"{method:<15} {'FAIL':>10} {'FAIL':>10} {'FAIL':>10} "
                  f"{'FAIL':>10} {'FAIL':>10} {'FAIL':>10} {0:>5}")

    # Save results
    save_results()
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()

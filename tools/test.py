"""Testing entry point."""

import argparse
import os
import sys

# Ensure repo root is importable when invoked as `python tools/test.py`
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault('HF_HUB_OFFLINE', '1')

# Match tools/train.py: allow forcing CPU before importing torch.
if '--cpu' in sys.argv:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTORCH_MPS_DISABLE'] = '1'

import torch

if '--cpu' in sys.argv:
    torch.backends.mps.is_available = lambda: False
    torch.backends.mps.is_built = lambda: False

# Match tools/train.py: force mmengine checkpoint loads to bypass
# PyTorch 2.6's weights_only=True default.
_original_torch_load = torch.load


def _torch_load_compat(f, map_location=None, pickle_module=None, *, weights_only=None, **kwargs):
    return _original_torch_load(
        f,
        map_location=map_location,
        pickle_module=pickle_module,
        weights_only=False,
        **kwargs,
    )


torch.load = _torch_load_compat

from mmengine.config import Config, DictAction
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description='Test an anomaly detector')
    parser.add_argument('config', help='Test config file path')
    parser.add_argument('checkpoint', nargs='?', default=None, help='Optional checkpoint file path')
    parser.add_argument('--work-dir', help='Working directory for results')
    parser.add_argument('--cpu', action='store_true', help='Force CPU device')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override config options in key=value format.',
    )
    return parser.parse_args()


def _apply_runtime_overrides(cfg: Config) -> None:
    disable_compile = bool(
        cfg.get('runtime_disable_compile', False)
        or cfg.get('train_disable_compile', False)
        or cfg.get('benchmark_disable_compile', False)
    )
    if not disable_compile:
        return

    os.environ['TORCH_COMPILE_DISABLE'] = '1'
    os.environ['TORCHDYNAMO_DISABLE'] = '1'
    for key in ['compile', 'compile_cfg', 'compile_options']:
        cfg.pop(key, None)
    model_wrapper_cfg = cfg.get('model_wrapper_cfg', None)
    if isinstance(model_wrapper_cfg, dict):
        wrapper_type = str(model_wrapper_cfg.get('type', ''))
        if 'compile' in wrapper_type.lower():
            cfg.pop('model_wrapper_cfg', None)


def main():
    args = parse_args()

    from baoiad import register_all_modules

    register_all_modules()

    cfg = Config.fromfile(args.config)
    from baoiad.config import apply_data_root_overrides

    apply_data_root_overrides(cfg)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    if args.work_dir:
        cfg.work_dir = args.work_dir
    if args.checkpoint:
        cfg.load_from = args.checkpoint
    _apply_runtime_overrides(cfg)

    runner = Runner.from_cfg(cfg)
    runner.test()


if __name__ == '__main__':
    main()

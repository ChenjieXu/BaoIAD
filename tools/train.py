"""Training entry point."""

import argparse
import os
import sys

# Ensure repo root is importable when invoked as `python tools/train.py`
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Prefer local HuggingFace/timm caches in offline lab environments.
os.environ.setdefault('HF_HUB_OFFLINE', '1')

# Check --cpu early before any torch import
if '--cpu' in sys.argv:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTORCH_MPS_DISABLE'] = '1'

import torch

# Monkey-patch torch.load for PyTorch 2.6+ compatibility with mmengine checkpoints
# mmengine checkpoints contain classes not allowed by weights_only=True
_original_torch_load = torch.load
def _torch_load_compat(f, map_location=None, pickle_module=None, *, weights_only=None, **kwargs):
    # Force weights_only=False for mmengine checkpoint compatibility
    return _original_torch_load(f, map_location=map_location, pickle_module=pickle_module, weights_only=False, **kwargs)
torch.load = _torch_load_compat

if '--cpu' in sys.argv:
    torch.backends.mps.is_available = lambda: False
    torch.backends.mps.is_built = lambda: False

from mmengine.config import Config, DictAction
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description='Train an anomaly detector')
    parser.add_argument('config', help='Train config file path')
    parser.add_argument('--work-dir', help='Working directory to save logs and models')
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
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

    import baoiad  # noqa: F401 - trigger registry

    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    if args.work_dir:
        cfg.work_dir = args.work_dir
    _apply_runtime_overrides(cfg)

    runner = Runner.from_cfg(cfg)
    if args.resume:
        # Resume from the latest checkpoint in work_dir
        resume_path = os.path.join(cfg.work_dir, 'last_checkpoint')
        if os.path.exists(resume_path):
            with open(resume_path, 'r') as f:
                checkpoint_path = f.read().strip()
            runner.resume(checkpoint_path)
    runner.train()


if __name__ == '__main__':
    main()

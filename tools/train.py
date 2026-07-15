"""Training entry point."""

import argparse
import os
import sys

# Ensure repo root is importable when invoked as `python tools/train.py`
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Check --cpu early before any torch import
if '--cpu' in sys.argv:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTORCH_MPS_DISABLE'] = '1'

from mmengine.config import Config, DictAction
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description='Train an anomaly detector')
    parser.add_argument('config', help='Train config file path')
    parser.add_argument('--work-dir', help='Working directory to save logs and models')
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    parser.add_argument('--cpu', action='store_true', help='Force CPU device')
    parser.add_argument(
        '--trusted-checkpoint',
        action='store_true',
        help='Allow legacy pickle checkpoints from a verified source (can execute code).',
    )
    parser.add_argument(
        '--offline',
        action='store_true',
        help='Disable model-hub and BaoIAD-managed downloads for this process.',
    )
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

    from baoiad.runtime import configure_offline_mode

    configure_offline_mode(args.offline)

    from baoiad import register_all_modules

    register_all_modules()

    cfg = Config.fromfile(args.config)
    from baoiad.config import apply_data_root_overrides

    apply_data_root_overrides(cfg)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    if args.work_dir:
        cfg.work_dir = args.work_dir
    _apply_runtime_overrides(cfg)

    from baoiad.checkpoint import checkpoint_loading_policy

    with checkpoint_loading_policy(args.trusted_checkpoint):
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

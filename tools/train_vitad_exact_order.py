#!/usr/bin/env python3
"""Train ViTAD with replayed official ADer train order."""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if '--cpu' in sys.argv:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTORCH_MPS_DISABLE'] = '1'

from mmengine.config import Config, DictAction
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description='Train ViTAD with exact official order replay.')
    parser.add_argument('config', help='Train config file path')
    parser.add_argument('--work-dir', help='Working directory to save logs and models')
    parser.add_argument(
        '--order-file',
        help='Verified per-epoch order JSON; BaoIAD does not generate or distribute it.',
    )
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


def _load_cfg(args) -> Config:
    cfg = Config.fromfile(args.config)
    from baoiad.config import apply_data_root_overrides

    apply_data_root_overrides(cfg)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    if args.work_dir:
        cfg.work_dir = args.work_dir
    elif 'work_dir' not in cfg:
        cfg.work_dir = str(ROOT / 'runs' / Path(args.config).stem)
    return cfg


def _configured_classes(cfg: Config) -> list[str]:
    classes = cfg.train_dataloader.dataset.get('cls_names')
    if classes:
        return list(classes)

    import baoiad  # noqa: F401
    from mmengine.registry import init_default_scope
    from baoiad.registry import DATASETS

    init_default_scope(cfg.get('default_scope', 'baoiad'))
    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    resolved = getattr(dataset, 'cls_names', None)
    if not resolved:
        raise ValueError('ViTAD exact-order training failed to infer cls_names from the dataset config.')
    return list(resolved)


def _order_file_path(cfg: Config, classes: list[str], epochs: int) -> Path:
    work_dir = Path(cfg.work_dir)
    if len(classes) == 15:
        class_tag = 'all15'
    else:
        class_tag = '_'.join(classes)
    return work_dir / f'official_order_{class_tag}_e{epochs}.json'


def _resolve_order_file(
    configured_path: str | None,
    cfg: Config,
    classes: list[str],
    epochs: int,
) -> Path:
    order_file = (
        Path(configured_path).expanduser()
        if configured_path
        else _order_file_path(cfg, classes, epochs)
    )
    order_file = order_file.resolve(strict=False)
    if not order_file.is_file():
        raise FileNotFoundError(
            'ViTAD exact-order replay requires a verified per-epoch order JSON. '
            f'No file was found at {order_file}. BaoIAD does not generate or distribute '
            'the official order artifact; obtain it under its source terms and pass '
            '--order-file PATH.'
        )
    return order_file


def _apply_exact_order_overrides(cfg: Config, order_file: Path) -> None:
    train_cfg = copy.deepcopy(cfg.train_cfg)
    train_cfg.pop('by_epoch', None)
    train_cfg['type'] = 'EpochBasedTrainLoop'
    cfg.train_cfg = train_cfg
    cfg.merge_from_dict({
        'train_dataloader.sampler.type': 'PerEpochOrderSampler',
        'train_dataloader.sampler.index_file': str(order_file),
        'train_dataloader.sampler.round_up': False,
    })


def main():
    args = parse_args()

    from baoiad.runtime import configure_offline_mode

    configure_offline_mode(args.offline)

    import baoiad  # noqa: F401

    cfg = _load_cfg(args)
    classes = _configured_classes(cfg)
    epochs = int(cfg.train_cfg.max_epochs)
    order_file = _resolve_order_file(args.order_file, cfg, classes, epochs)
    _apply_exact_order_overrides(cfg, order_file)

    from baoiad.checkpoint import checkpoint_loading_policy

    with checkpoint_loading_policy(args.trusted_checkpoint):
        runner = Runner.from_cfg(cfg)
        if args.resume:
            resume_path = Path(cfg.work_dir) / 'last_checkpoint'
            if resume_path.exists():
                checkpoint_path = resume_path.read_text().strip()
                runner.resume(checkpoint_path)
        runner.train()


if __name__ == '__main__':
    main()

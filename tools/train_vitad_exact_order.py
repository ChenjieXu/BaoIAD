#!/usr/bin/env python3
"""Train ViTAD with replayed official ADer train order."""

from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if '--cpu' in sys.argv:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTORCH_MPS_DISABLE'] = '1'

import torch

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

if '--cpu' in sys.argv:
    torch.backends.mps.is_available = lambda: False
    torch.backends.mps.is_built = lambda: False

from mmengine.config import Config, DictAction
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description='Train ViTAD with exact official order replay.')
    parser.add_argument('config', help='Train config file path')
    parser.add_argument('--work-dir', help='Working directory to save logs and models')
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    parser.add_argument('--cpu', action='store_true', help='Force CPU device')
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

    import iadbench  # noqa: F401
    from mmengine.registry import init_default_scope
    from iadbench.registry import DATASETS

    init_default_scope(cfg.get('default_scope', 'iadbench'))
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


def _dump_official_order(config_path: str, classes: list[str], epochs: int, output_path: Path) -> None:
    if output_path.exists():
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / 'tools' / 'vitad_dump_official_order.py'),
        str(Path(config_path).resolve()),
        '--classes',
        *classes,
        '--epochs',
        str(int(epochs)),
        '--output',
        str(output_path),
    ]
    subprocess.run(cmd, cwd=ROOT, env=dict(os.environ), check=True)


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

    import iadbench  # noqa: F401

    cfg = _load_cfg(args)
    classes = _configured_classes(cfg)
    epochs = int(cfg.train_cfg.max_epochs)
    order_file = _order_file_path(cfg, classes, epochs)
    _dump_official_order(args.config, classes, epochs, order_file)
    _apply_exact_order_overrides(cfg, order_file)

    runner = Runner.from_cfg(cfg)
    if args.resume:
        resume_path = Path(cfg.work_dir) / 'last_checkpoint'
        if resume_path.exists():
            checkpoint_path = resume_path.read_text().strip()
            runner.resume(checkpoint_path)
    runner.train()


if __name__ == '__main__':
    main()

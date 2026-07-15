#!/usr/bin/env python3
"""Official-compatible RegAD training and evaluation entry point."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

if '--cpu' in sys.argv:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTORCH_MPS_DISABLE'] = '1'

import torch
import numpy as np

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
from mmengine.registry import init_default_scope
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description='Train strict RegAD.')
    parser.add_argument('config', help='Train config file path')
    parser.add_argument('--work-dir', help='Working directory to save logs and checkpoints')
    parser.add_argument('--resume', action='store_true', help='Resume from `last_checkpoint` if present')
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


def _scalar(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _load_cfg(args) -> Config:
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    if args.work_dir:
        cfg.work_dir = args.work_dir
    return cfg


def _device_from_args(args) -> torch.device:
    if args.cpu or not torch.cuda.is_available():
        return torch.device('cpu')
    return torch.device('cuda')


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, base_lr: float, epoch: int, max_epochs: int) -> None:
    lr = base_lr * 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * epoch / max_epochs)).item())
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def _extract_gt_label(sample) -> int:
    value = sample.gt_label
    if torch.is_tensor(value):
        return int(value.detach().cpu().item())
    return int(value)


def _extract_gt_mask(sample):
    value = sample.gt_mask
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return value


def _save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def _write_metrics_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as handle:
        json.dump(payload, handle, indent=2)


def _resume_if_needed(cfg: Config, args, model, optimizer, device: torch.device):
    if not args.resume:
        return 1, {'image_auroc': float('-inf'), 'pixel_auroc': float('-inf')}, float('-inf')

    last_pointer = Path(cfg.work_dir) / 'last_checkpoint'
    if not last_pointer.is_file():
        return 1, {'image_auroc': float('-inf'), 'pixel_auroc': float('-inf')}, float('-inf')

    checkpoint_path = Path(last_pointer.read_text().strip())
    if not checkpoint_path.is_file():
        return 1, {'image_auroc': float('-inf'), 'pixel_auroc': float('-inf')}, float('-inf')

    state = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    best_metrics = state.get('best_metrics', {'image_auroc': float('-inf'), 'pixel_auroc': float('-inf')})
    best_score = state.get('best_score', float('-inf'))
    return int(state.get('epoch', 0)) + 1, best_metrics, float(best_score)


def main():
    args = parse_args()

    from baoiad.runtime import configure_offline_mode

    configure_offline_mode(args.offline)

    import iadbench  # noqa: F401
    from iadbench.registry import MODELS
    from iadbench.utils.alignment_probe import (
        move_data_samples_to_device,
        move_inputs_to_device,
        set_global_seed,
    )
    from iadbench.utils.regad_strict import compute_regad_metrics, load_or_sample_support_rounds

    cfg = _load_cfg(args)
    init_default_scope('iadbench')

    work_dir = Path(cfg.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    device = _device_from_args(args)
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    seed = int(cfg.get('official_seed', cfg.get('randomness', {}).get('seed', 668)))
    set_global_seed(seed)
    print(f'[RegAD] device={device} seed={seed}', flush=True)

    model = MODELS.build(cfg.model).to(device)
    model.target_cls = cfg.model.get('target_cls', cfg.get('target_cls', None))
    model.data_root = cfg.model.get('data_root', cfg.get('data_root', None))

    optimizer_cfg = dict(cfg.optim_wrapper.optimizer)
    optimizer_type = optimizer_cfg.pop('type')
    if optimizer_type != 'SGD':
        raise ValueError(f'RegAD strict only supports SGD optimizer, got {optimizer_type!r}')
    base_lr = float(optimizer_cfg.pop('lr'))
    optimizer = torch.optim.SGD(
        [param for param in model.parameters() if param.requires_grad],
        lr=base_lr,
        **optimizer_cfg,
    )

    start_epoch, best_metrics, best_score = _resume_if_needed(cfg, args, model, optimizer, device)

    max_epochs = int(cfg.train_cfg.max_epochs)
    target_cls = cfg.model.get('target_cls', cfg.test_dataloader.dataset.get('target_cls'))
    data_root = cfg.model.get('data_root', cfg.test_dataloader.dataset.get('data_root'))
    img_size = int(cfg.get('img_size', cfg.model.get('img_size', 224)))
    shot = int(cfg.get('shot', cfg.train_dataloader.dataset.get('shot', cfg.model.get('few_shot', 4))))
    inferences = int(cfg.get('inferences', 10))
    support_set_root = cfg.get('support_set_root', None)
    require_official_support_set = bool(cfg.get('strict_require_official_support_set', False))

    support_rounds, support_round_source, support_round_file = load_or_sample_support_rounds(
        data_root=data_root,
        target_cls=target_cls,
        img_size=img_size,
        shot=shot,
        inferences=inferences,
        seed=seed,
        support_set_root=support_set_root,
        allow_fallback=not require_official_support_set,
    )
    print(f'[RegAD] support_rounds={len(support_rounds)} shot={shot} img_size={img_size}', flush=True)
    if support_round_source == 'official':
        print(f'[RegAD] support_set_file={support_round_file}', flush=True)
    elif support_set_root:
        print(
            f'[RegAD] support_set_root={support_set_root} missing {support_round_file}; '
            'using deterministic local support sampling fallback.',
            flush=True,
        )
    else:
        print('[RegAD] support_set_root missing; using deterministic local support sampling fallback.', flush=True)

    test_loader = Runner.build_dataloader(cfg.test_dataloader, seed=seed)

    def evaluate_current_model():
        eval_start = time.time()
        round_metrics = []
        model.eval()
        with torch.inference_mode():
            for support_images in support_rounds:
                model.build_support_bank_from_images(support_images)
                score_maps = []
                gt_labels = []
                gt_masks = []
                for batch in test_loader:
                    inputs = move_inputs_to_device(batch['inputs'], device)
                    data_samples = move_data_samples_to_device(batch.get('data_samples', []), device)
                    anomaly_map, _ = model.predict_raw_maps(inputs)
                    score_maps.append(anomaly_map.squeeze(1).detach().cpu().numpy())
                    gt_labels.extend(_extract_gt_label(sample) for sample in data_samples)
                    gt_masks.extend(_extract_gt_mask(sample) for sample in data_samples)

                metrics = compute_regad_metrics(
                    score_maps=np.concatenate(score_maps, axis=0),
                    gt_labels=gt_labels,
                    gt_masks=np.stack(gt_masks, axis=0),
                )
                round_metrics.append(metrics)

        image_auroc = sum(metric['image_auroc'] for metric in round_metrics) / len(round_metrics)
        pixel_auroc = sum(metric['pixel_auroc'] for metric in round_metrics) / len(round_metrics)
        print(f'[RegAD] eval_time={time.time() - eval_start:.1f}s', flush=True)
        return {
            'image_auroc': image_auroc,
            'pixel_auroc': pixel_auroc,
        }

    def train_one_epoch(epoch_seed: int):
        train_start = time.time()
        set_global_seed(epoch_seed)
        train_loader = Runner.build_dataloader(cfg.train_dataloader, seed=epoch_seed)
        model.train()
        total_loss = 0.0
        total_samples = 0
        for batch in train_loader:
            inputs = move_inputs_to_device(batch['inputs'], device)
            data_samples = move_data_samples_to_device(batch.get('data_samples', []), device)
            optimizer.zero_grad(set_to_none=True)
            loss_dict = model(inputs, data_samples, mode='loss')
            loss = loss_dict['loss']
            loss.backward()
            optimizer.step()

            batch_size = int(inputs.shape[0]) if torch.is_tensor(inputs) else len(data_samples)
            total_loss += _scalar(loss) * batch_size
            total_samples += batch_size
        print(f'[RegAD] train_time={time.time() - train_start:.1f}s', flush=True)
        return total_loss / max(total_samples, 1)

    for epoch in range(start_epoch, max_epochs + 1):
        _set_optimizer_lr(optimizer, base_lr, epoch, max_epochs)
        eval_metrics = evaluate_current_model()
        current_score = eval_metrics['image_auroc'] + eval_metrics['pixel_auroc']

        print(
            f"Epoch {epoch:03d} ad/image_auroc: {eval_metrics['image_auroc']:.4f} "
            f"ad/pixel_auroc: {eval_metrics['pixel_auroc']:.4f}",
            flush=True,
        )
        _write_metrics_json(
            work_dir / 'latest_metrics.json',
            {
                'epoch': epoch,
                'metrics': eval_metrics,
            },
        )

        best_payload = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_metrics': best_metrics,
            'best_score': best_score,
        }
        if current_score > best_score:
            best_score = current_score
            best_metrics = dict(eval_metrics)
            best_payload['best_metrics'] = best_metrics
            best_payload['best_score'] = best_score
            _save_checkpoint(work_dir / 'best_balanced.pth', best_payload)
            _write_metrics_json(
                work_dir / 'best_metrics.json',
                {
                    'epoch': epoch,
                    'metrics': best_metrics,
                    'best_score': best_score,
                },
            )

        epoch_loss = train_one_epoch(seed + epoch)
        last_checkpoint = work_dir / 'last.pth'
        _save_checkpoint(
            last_checkpoint,
            {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_metrics': best_metrics,
                'best_score': best_score,
            },
        )
        (work_dir / 'last_checkpoint').write_text(str(last_checkpoint))
        print(
            f"[RegAD] epoch={epoch:03d} loss={epoch_loss:.6f} "
            f"best_image_auroc={best_metrics['image_auroc']:.4f} "
            f"best_pixel_auroc={best_metrics['pixel_auroc']:.4f}",
            flush=True,
        )


if __name__ == '__main__':
    main()

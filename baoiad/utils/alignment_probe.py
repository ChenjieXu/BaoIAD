"""Alignment probe helpers and CLI implementation."""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import random
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Sequence

import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.runner import Runner


def set_global_seed(seed: int) -> None:
    """Set Python / NumPy / Torch RNG state for reproducible diagnostics."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _tensor_stats(tensor: torch.Tensor | np.ndarray | None) -> Dict[str, Any] | None:
    """Summarize a tensor-like object with shape/range statistics."""
    if tensor is None:
        return None
    if not torch.is_tensor(tensor):
        tensor = torch.as_tensor(tensor)

    detached = tensor.detach().cpu()
    if detached.numel() == 0:
        return {
            'shape': list(detached.shape),
            'dtype': str(detached.dtype),
            'numel': 0,
            'finite': True,
            'min': None,
            'max': None,
            'mean': None,
            'std': None,
        }

    finite = torch.isfinite(detached).all().item()
    float_tensor = detached.float()
    return {
        'shape': list(detached.shape),
        'dtype': str(detached.dtype),
        'numel': int(detached.numel()),
        'finite': bool(finite),
        'min': float(float_tensor.min().item()),
        'max': float(float_tensor.max().item()),
        'mean': float(float_tensor.mean().item()),
        'std': float(float_tensor.std(unbiased=False).item()),
    }


def _json_scalar(value: Any) -> Any:
    """Convert a scalar-like object into a JSON-serializable value."""
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value) and value.ndim == 0:
        return value.detach().cpu().item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _summarize_value(value: Any, *, max_items: int = 8) -> Any:
    """Summarize an arbitrary value for JSON-friendly probe output."""
    if torch.is_tensor(value) or isinstance(value, np.ndarray):
        return _tensor_stats(value)
    if isinstance(value, Mapping):
        summary: Dict[str, Any] = {}
        for idx, (key, item) in enumerate(value.items()):
            if idx >= max_items:
                summary['...'] = f'{len(value) - max_items} more items'
                break
            summary[str(key)] = _summarize_value(item, max_items=max_items)
        return summary
    if isinstance(value, (list, tuple)):
        return {
            'type': type(value).__name__,
            'length': len(value),
            'items': [_summarize_value(item, max_items=max_items) for item in value[:max_items]],
        }
    return _json_scalar(value)


def _prepare_dataloader_cfg(
    dataloader_cfg: MutableMapping[str, Any],
    max_batch_size: int,
) -> MutableMapping[str, Any]:
    """Clone a dataloader config and clamp it for lightweight probe runs."""
    cfg = copy.deepcopy(dict(dataloader_cfg))
    cfg['batch_size'] = min(int(cfg.get('batch_size', max_batch_size)), max_batch_size)
    cfg['num_workers'] = 0
    cfg['persistent_workers'] = False
    _inject_default_single_class_target(cfg)
    return cfg


def _inject_default_single_class_target(
    dataloader_cfg: MutableMapping[str, Any],
    *,
    default_cls_name: str = 'bottle',
) -> None:
    """Inject a deterministic default class for single-class probe configs."""

    def _visit(dataset_cfg: Any) -> None:
        if not isinstance(dataset_cfg, MutableMapping):
            return
        nested = dataset_cfg.get('dataset')
        if nested is not None:
            _visit(nested)

        if dataset_cfg.get('multi_class') is False and not dataset_cfg.get('cls_names'):
            dataset_cfg['cls_names'] = [default_cls_name]

    dataset_cfg = dataloader_cfg.get('dataset')
    if dataset_cfg is not None:
        _visit(dataset_cfg)


def ensure_data_samples_list(data_samples: Any) -> list[Any]:
    """Normalize data_samples into a list."""
    if data_samples is None:
        return []
    if isinstance(data_samples, list):
        return data_samples
    if isinstance(data_samples, tuple):
        return list(data_samples)
    return [data_samples]


def move_inputs_to_device(inputs: Any, device: torch.device) -> Any:
    """Move tensor-like inputs onto the target device."""
    return _move_inputs_to_device(inputs, device)


def move_data_samples_to_device(data_samples: Sequence[Any], device: torch.device) -> list[Any]:
    """Move MMEngine data samples onto the target device when supported."""
    return _move_data_samples_to_device(data_samples, device)


def _move_inputs_to_device(inputs: Any, device: torch.device) -> Any:
    if torch.is_tensor(inputs):
        return inputs.to(device)
    if isinstance(inputs, list):
        return [item.to(device) if torch.is_tensor(item) else item for item in inputs]
    if isinstance(inputs, tuple):
        return tuple(item.to(device) if torch.is_tensor(item) else item for item in inputs)
    return inputs


def _move_data_samples_to_device(data_samples: Sequence[Any], device: torch.device) -> list[Any]:
    moved = []
    for sample in data_samples:
        if hasattr(sample, 'to'):
            moved.append(sample.to(device))
        else:
            moved.append(sample)
    return moved


def _stack_inputs(inputs: Any) -> torch.Tensor | None:
    if torch.is_tensor(inputs):
        return inputs
    if isinstance(inputs, (list, tuple)) and inputs and all(torch.is_tensor(item) for item in inputs):
        return torch.stack(list(inputs))
    return None


def summarize_inputs(inputs: Any) -> Dict[str, Any]:
    """Summarize model inputs using a compact public JSON shape."""
    if torch.is_tensor(inputs) or isinstance(inputs, np.ndarray):
        return {'kind': 'tensor', 'value': _tensor_stats(inputs)}
    if isinstance(inputs, list):
        return {
            'kind': 'list',
            'length': len(inputs),
            'items': [_summarize_value(item) for item in inputs[:4]],
        }
    if isinstance(inputs, tuple):
        return {
            'kind': 'tuple',
            'length': len(inputs),
            'items': [_summarize_value(item) for item in inputs[:4]],
        }
    return {'kind': type(inputs).__name__, 'value': _summarize_value(inputs)}


def _summarize_inputs(inputs: Any) -> Dict[str, Any]:
    stacked = _stack_inputs(inputs)
    if torch.is_tensor(inputs):
        return {
            'container': 'tensor',
            'batch_size': int(inputs.shape[0]),
            'stats': _tensor_stats(inputs),
        }

    if isinstance(inputs, (list, tuple)):
        summary = {
            'container': 'sequence',
            'length': len(inputs),
            'batch_size': len(inputs),
            'shapes': [list(item.shape) if torch.is_tensor(item) else str(type(item)) for item in inputs],
        }
        if inputs:
            first = inputs[0]
            if torch.is_tensor(first):
                summary['first'] = _tensor_stats(first)
        if stacked is not None:
            summary['stacked'] = _tensor_stats(stacked)
            summary['finite'] = bool(torch.isfinite(stacked).all().item())
        return summary

    return {
        'container': type(inputs).__name__,
        'batch_size': None,
    }


def summarize_data_samples(data_samples: Any, *, max_samples: int = 2) -> Dict[str, Any]:
    """Summarize data-sample fields and metainfo for probe output."""
    samples = ensure_data_samples_list(data_samples)
    metainfo_keys = set()
    data_keys = set()
    preview = []

    for sample in samples:
        if hasattr(sample, 'metainfo_keys'):
            metainfo_keys.update(sample.metainfo_keys())
        if hasattr(sample, 'keys'):
            data_keys.update(sample.keys())

    for sample in samples[:max_samples]:
        metainfo = {}
        if hasattr(sample, 'metainfo_items'):
            metainfo = {
                str(key): _summarize_value(value)
                for key, value in sample.metainfo_items()
            }
        fields = {}
        if hasattr(sample, 'items'):
            fields = {
                str(key): _summarize_value(value)
                for key, value in sample.items()
            }
        preview.append({
            'metainfo': metainfo,
            'fields': fields,
        })

    return {
        'count': len(samples),
        'metainfo_keys': sorted(str(key) for key in metainfo_keys),
        'data_keys': sorted(str(key) for key in data_keys),
        'preview': preview,
    }


def _data_sample_preview(data_sample: Any) -> Dict[str, Any]:
    preview = {
        'type': type(data_sample).__name__,
        'metainfo_keys': [],
        'metainfo': {},
        'keys': [],
        'fields': {},
    }
    if data_sample is None:
        return preview

    if hasattr(data_sample, 'metainfo_keys'):
        preview['metainfo_keys'] = sorted(list(data_sample.metainfo_keys()))
    if hasattr(data_sample, 'metainfo_items'):
        preview['metainfo'] = {
            str(key): _summarize_value(value)
            for key, value in data_sample.metainfo_items()
        }

    for key in ['cls_name', 'defect_type', 'img_path', 'gt_label', 'gt_mask']:
        if not hasattr(data_sample, key):
            continue
        value = getattr(data_sample, key)
        preview['keys'].append(key)
        preview['fields'][key] = _summarize_value(value)

    return preview


def _summarize_data_sample(data_sample: Any) -> Dict[str, Any]:
    """Return a flat per-sample summary for CLI/debug helpers."""
    preview = _data_sample_preview(data_sample)
    summary = dict(preview.get('fields', {}))
    if preview.get('metainfo'):
        summary['metainfo'] = preview['metainfo']
    return summary


def _label_counts(data_samples: Sequence[Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for sample in data_samples:
        label = None
        if hasattr(sample, 'gt_label'):
            value = getattr(sample, 'gt_label')
            if torch.is_tensor(value):
                label = int(value.item())
            else:
                label = int(value)
        if label is None:
            continue
        key = str(label)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _add_check(report: Dict[str, Any], name: str, ok: bool, detail: str) -> None:
    report['checks'].append({
        'name': name,
        'ok': bool(ok),
        'detail': detail,
    })


def _call_optional_builder(method: Any, train_loader: Any) -> int:
    signature = inspect.signature(method)
    if signature.parameters:
        method(train_loader)
        return 1 if train_loader is not None else 0
    method()
    return 0


def _staged_memory_bank_size(owner: Any) -> int:
    memory_bank = getattr(owner, '_memory_bank', None)
    if memory_bank is None:
        return 0
    if torch.is_tensor(memory_bank):
        return int(memory_bank.shape[0]) if memory_bank.ndim > 0 else 0
    if isinstance(memory_bank, list):
        total = 0
        for item in memory_bank:
            if torch.is_tensor(item) and item.ndim > 0:
                total += int(item.shape[0])
        return total
    return 0


def _required_memory_bank_samples(model: Any) -> int:
    classifier = getattr(model, 'classifier', None)
    required = getattr(classifier, 'n_pca_components', None)
    if required is None:
        return 0
    required = int(required)
    return max(required + 1, required * 2)


def _collect_loss_batches_for_probe(
    model: Any,
    train_loader: Any,
    device: torch.device,
    min_samples: int,
) -> int:
    if train_loader is None or min_samples <= 0:
        return 0

    staged_samples = _staged_memory_bank_size(model)
    if staged_samples >= min_samples:
        return 0

    collected_batches = 0
    model.train()
    with torch.no_grad():
        for batch in train_loader:
            inputs, data_samples = _extract_batch_fields(batch)
            moved_inputs = _move_inputs_to_device(inputs, device)
            moved_samples = _move_data_samples_to_device(data_samples, device)
            model(moved_inputs, moved_samples, mode='loss')
            collected_batches += 1
            staged_samples = _staged_memory_bank_size(model)
            if staged_samples >= min_samples:
                break
    model.eval()
    return collected_batches


def warmup_memory_bank_for_probe(model: Any, train_loader: Any, device: torch.device) -> Dict[str, Any]:
    """Try the standard memory-bank builders used by probe-capable methods."""
    target_samples = _required_memory_bank_samples(model)
    loss_batches = _collect_loss_batches_for_probe(model, train_loader, device, target_samples)
    info = {
        'used': False,
        'num_batches': 0,
        'loss_batches': loss_batches,
        'staged_samples': _staged_memory_bank_size(model),
        'target_samples': target_samples,
    }

    if hasattr(model, 'build_template_from_dataloader') and train_loader is not None:
        model.build_template_from_dataloader(train_loader, device)
        info['used'] = True
        info['builder'] = 'build_template_from_dataloader'
        return info

    for owner in (model, getattr(model, 'head', None)):
        if owner is None:
            continue
        for method_name in ('build_memory_bank', 'fit'):
            if not hasattr(owner, method_name):
                continue
            method = getattr(owner, method_name)
            try:
                with torch.inference_mode(False), torch.enable_grad():
                    info['num_batches'] = _call_optional_builder(method, train_loader)
            except RuntimeError as exc:
                if train_loader is None or 'positive-definite' not in str(exc):
                    raise
                retry_target = max(info['target_samples'] * 2, _staged_memory_bank_size(model) + info['target_samples'])
                info['loss_batches'] += _collect_loss_batches_for_probe(model, train_loader, device, retry_target)
                info['retry_error'] = f'{type(exc).__name__}: {exc}'
                with torch.inference_mode(False), torch.enable_grad():
                    info['num_batches'] = _call_optional_builder(method, train_loader)
            info['used'] = True
            info['builder'] = method_name
            return info
    return info


def _resolve_device(device: str) -> torch.device:
    if device == 'cuda':
        return torch.device('cuda')
    if device == 'cpu':
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def _load_first_batch(dataloader: Any) -> Any:
    iterator = iter(dataloader)
    return next(iterator)


def _extract_batch_fields(batch: Any) -> tuple[Any, list[Any]]:
    if isinstance(batch, dict):
        inputs = batch.get('inputs', batch.get('img'))
        data_samples = list(batch.get('data_samples', []))
        return inputs, data_samples
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        inputs = batch[0]
        data_samples = list(batch[1]) if isinstance(batch[1], (list, tuple)) else []
        return inputs, data_samples
    return batch, []


def _summarize_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Backward-compatible batch summary used by CLI tests."""
    inputs, data_samples = _extract_batch_fields(batch)
    if isinstance(inputs, (list, tuple)):
        input_stats = [_tensor_stats(item) for item in inputs]
    else:
        input_stats = _tensor_stats(inputs)
    return {
        'inputs': input_stats,
        'num_samples': len(data_samples),
        'samples': [_summarize_data_sample(sample) for sample in data_samples[:2]],
    }


def _summarize_predict_outputs(outputs: Sequence[Any]) -> Dict[str, Any]:
    score_values = []
    maps = []
    for output in outputs:
        if hasattr(output, 'pred_score'):
            score_values.append(float(getattr(output, 'pred_score')))
        if hasattr(output, 'pred_anomaly_map'):
            maps.append(getattr(output, 'pred_anomaly_map'))

    score_tensor = torch.tensor(score_values, dtype=torch.float32) if score_values else torch.empty(0)
    map_tensor = None
    if maps and all(torch.is_tensor(item) for item in maps):
        map_tensor = torch.stack([item.detach().cpu() for item in maps])

    return {
        'num_results': len(outputs),
        'score_stats': _tensor_stats(score_tensor),
        'maps_present': len(maps) == len(outputs),
        'map_shapes': [list(map_item.shape) for map_item in maps if torch.is_tensor(map_item)],
        'map_stats': _tensor_stats(map_tensor) if map_tensor is not None else None,
    }


def summarize_losses(losses: Mapping[str, Any]) -> Dict[str, Any]:
    """Summarize loss dicts returned by ``mode='loss'``."""
    summary = {
        str(key): _summarize_value(value)
        for key, value in dict(losses).items()
    }
    finite = True
    for value in losses.values():
        if torch.is_tensor(value):
            finite = finite and bool(torch.isfinite(value.detach()).all().item())
    summary['all_tensor_values_finite'] = finite
    return summary


def summarize_predictions(outputs: Any, *, max_samples: int = 2) -> Dict[str, Any]:
    """Summarize predictions returned by ``mode='predict'``."""
    samples = ensure_data_samples_list(outputs)
    score_values = []
    preview = []

    for sample in samples[:max_samples]:
        item: Dict[str, Any] = {}
        if hasattr(sample, 'pred_score'):
            item['pred_score'] = float(sample.pred_score)
            score_values.append(float(sample.pred_score))
        if hasattr(sample, 'pred_anomaly_map'):
            item['pred_anomaly_map'] = _tensor_stats(sample.pred_anomaly_map)
        if hasattr(sample, 'gt_mask'):
            item['gt_mask'] = _tensor_stats(sample.gt_mask)
        if hasattr(sample, 'metainfo_items'):
            item['metainfo'] = {
                str(key): _summarize_value(value)
                for key, value in sample.metainfo_items()
            }
        preview.append(item)

    if len(samples) > max_samples:
        for sample in samples[max_samples:]:
            if hasattr(sample, 'pred_score'):
                score_values.append(float(sample.pred_score))

    score_stats = None
    if score_values:
        score_stats = _tensor_stats(torch.tensor(score_values, dtype=torch.float32))

    return {
        'count': len(samples),
        'pred_score_stats': score_stats,
        'preview': preview,
    }


def probe_config(
    config_path: str,
    splits: Sequence[str] = ('train', 'test'),
    max_batch_size: int = 2,
    device: str = 'auto',
    seed: int | None = None,
    cfg_options: dict[str, Any] | None = None,
    output: str | None = None,
    offline: bool = False,
    trusted_checkpoint: bool = False,
) -> Dict[str, Any]:
    """Run a lightweight structural probe under a scoped checkpoint policy."""
    from baoiad.checkpoint import checkpoint_loading_policy

    with checkpoint_loading_policy(trusted_checkpoint):
        return _probe_config_impl(
            config_path=config_path,
            splits=splits,
            max_batch_size=max_batch_size,
            device=device,
            seed=seed,
            cfg_options=cfg_options,
            output=output,
            offline=offline,
        )


def _probe_config_impl(
    config_path: str,
    splits: Sequence[str] = ('train', 'test'),
    max_batch_size: int = 2,
    device: str = 'auto',
    seed: int | None = None,
    cfg_options: dict[str, Any] | None = None,
    output: str | None = None,
    offline: bool = False,
) -> Dict[str, Any]:
    """Implement the probe while the caller owns checkpoint policy scope."""
    from baoiad.runtime import configure_offline_mode

    configure_offline_mode(offline)

    import baoiad  # noqa: F401
    from baoiad.registry import MODELS

    cfg = Config.fromfile(config_path)
    from baoiad.config import apply_data_root_overrides

    apply_data_root_overrides(cfg)
    if cfg_options:
        cfg.merge_from_dict(cfg_options)

    if seed is None:
        seed = int(cfg.get('randomness', {}).get('seed', 42))

    set_global_seed(seed)
    torch_device = _resolve_device(device)
    init_default_scope(cfg.get('default_scope', 'baoiad'))

    report: Dict[str, Any] = {
        'checks': [],
        'device': str(torch_device),
        'max_batch_size': max_batch_size,
        'model_type': cfg.model.get('type', 'unknown'),
        'passed': False,
        'seed': seed,
        'splits': {},
    }

    model = MODELS.build(cfg.model).to(torch_device)

    train_loader = None
    if hasattr(cfg, 'train_dataloader'):
        train_loader = Runner.build_dataloader(
            _prepare_dataloader_cfg(cfg.train_dataloader, max_batch_size),
            seed=seed,
        )

    if hasattr(model, 'pre_train_setup') and train_loader is not None:
        model.pre_train_setup(train_loader)
    if hasattr(model, 'prepare_strict_epoch') and train_loader is not None:
        model.prepare_strict_epoch(train_loader)

    for split in splits:
        dataloader_cfg = getattr(cfg, f'{split}_dataloader')
        dataloader = train_loader if split == 'train' and train_loader is not None else Runner.build_dataloader(
            _prepare_dataloader_cfg(dataloader_cfg, max_batch_size),
            seed=seed,
        )
        batch = _load_first_batch(dataloader)
        inputs, data_samples = _extract_batch_fields(batch)
        inputs_summary = _summarize_inputs(inputs)
        batch_size = inputs_summary.get('batch_size')
        num_samples = len(data_samples)

        split_report: Dict[str, Any] = {
            'batch': {
                'batch_size': batch_size,
                'inputs': inputs_summary,
                'num_data_samples': num_samples,
                'label_counts': _label_counts(data_samples),
                'sample_preview': _data_sample_preview(data_samples[0]) if data_samples else None,
            }
        }
        report['splits'][split] = split_report

        _add_check(report, f'{split}.batch.non_empty', batch_size is not None and batch_size > 0, f'batch_size={batch_size}')
        _add_check(
            report,
            f'{split}.batch.sample_count_matches',
            batch_size is None or num_samples == 0 or batch_size == num_samples,
            f'inputs={batch_size}, data_samples={num_samples}',
        )

        stacked_inputs = _stack_inputs(inputs)
        inputs_finite = True
        if stacked_inputs is not None:
            inputs_finite = bool(torch.isfinite(stacked_inputs).all().item())
        _add_check(report, f'{split}.batch.inputs_finite', inputs_finite, 'all input tensors are finite')

        moved_inputs = _move_inputs_to_device(inputs, torch_device)
        moved_samples = _move_data_samples_to_device(data_samples, torch_device)

        if split == 'train':
            model.train()
            loss_outputs = model(moved_inputs, moved_samples, mode='loss')
            loss_values = {
                key: _tensor_stats(value)
                for key, value in loss_outputs.items()
                if torch.is_tensor(value)
            } if isinstance(loss_outputs, dict) else {}
            all_finite = all(value.get('finite', False) for value in loss_values.values())
            split_report['loss'] = {
                'keys': sorted(list(loss_outputs.keys())) if isinstance(loss_outputs, dict) else [],
                'all_finite': all_finite,
                'values': loss_values,
            }
            _add_check(report, 'train.loss.is_dict', isinstance(loss_outputs, dict), f'type={type(loss_outputs).__name__}')
            _add_check(
                report,
                'train.loss.non_empty',
                isinstance(loss_outputs, dict) and bool(loss_outputs),
                f"keys={sorted(list(loss_outputs.keys())) if isinstance(loss_outputs, dict) else []}",
            )
            _add_check(report, 'train.loss.all_finite', all_finite, 'all loss terms are finite')
            continue

        if hasattr(model, 'compute_normalization_stats'):
            model.compute_normalization_stats(dataloader)
            split_report['normalization_warmup'] = {
                'used': True,
            }

        model.eval()
        predict_error = None
        with torch.inference_mode():
            try:
                outputs = model(moved_inputs, moved_samples, mode='predict')
            except Exception as exc:  # noqa: BLE001 - probe should capture the trigger path
                predict_error = exc
                warmup_info = {
                    'trigger_error': f'{type(exc).__name__}: {exc}',
                }
                warmup_info.update(warmup_memory_bank_for_probe(model, train_loader, torch_device))
                split_report['memory_bank_warmup'] = warmup_info
                outputs = model(moved_inputs, moved_samples, mode='predict')

        output_summary = _summarize_predict_outputs(outputs)
        split_report['predict'] = output_summary

        num_results = output_summary['num_results']
        _add_check(
            report,
            f'{split}.predict.result_count_matches',
            batch_size is None or num_results == batch_size,
            f'results={num_results}, batch_size={batch_size}',
        )
        score_finite = bool(output_summary['score_stats'] and output_summary['score_stats'].get('finite', False))
        _add_check(report, f'{split}.predict.scores_finite', score_finite, 'all pred_score values are finite')
        _add_check(
            report,
            f'{split}.predict.maps_present',
            bool(output_summary['maps_present']),
            'all results expose pred_anomaly_map',
        )
        maps_finite = bool(output_summary['map_stats'] and output_summary['map_stats'].get('finite', False))
        _add_check(report, f'{split}.predict.maps_finite', maps_finite, 'all pred_anomaly_map values are finite')
        if predict_error is not None and 'memory_bank_warmup' not in split_report:
            split_report['memory_bank_warmup'] = {
                'used': False,
                'num_batches': 0,
                'trigger_error': f'{type(predict_error).__name__}: {predict_error}',
            }

    report['passed'] = all(check['ok'] for check in report['checks'])

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2))

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run a lightweight alignment probe')
    parser.add_argument('config', help='Config file path')
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'test'],
        choices=['train', 'val', 'test'],
        help='Which dataloader splits to probe',
    )
    parser.add_argument('--max-batch-size', type=int, default=2, help='Clamp dataloader batch size')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto', help='Probe device')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for probe (defaults to config randomness.seed)')
    parser.add_argument('--output', help='Optional JSON output path')
    parser.add_argument(
        '--offline',
        action='store_true',
        help='Disable model-hub and BaoIAD-managed downloads for this process.',
    )
    parser.add_argument(
        '--trusted-checkpoint',
        action='store_true',
        help='Allow legacy pickle checkpoints from a verified source (can execute code).',
    )
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override config options in key=value format.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = probe_config(
        config_path=args.config,
        splits=args.splits,
        max_batch_size=args.max_batch_size,
        device=args.device,
        seed=args.seed,
        cfg_options=args.cfg_options,
        output=args.output,
        offline=args.offline,
        trusted_checkpoint=args.trusted_checkpoint,
    )
    print(json.dumps(report, indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

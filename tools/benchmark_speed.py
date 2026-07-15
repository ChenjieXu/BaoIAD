#!/usr/bin/env python
"""Benchmark inference speed for BaoIAD models using real MVTec AD images.

Usage:
    python tools/benchmark_speed.py --methods patchcore,padim,rd --gpu 0 --output results/speed_patchcore.json

For each method: load model, pick 1 normal + 1 anomalous image per MVTec category,
warmup, then measure forward-pass latency.
"""

import argparse
import glob
import json
import os
import sys
import time
import traceback

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from baoiad import register_all_modules  # noqa: E402

register_all_modules()
from mmengine.config import Config  # noqa: E402
from baoiad.config import apply_data_root_overrides  # noqa: E402
from baoiad.registry import MODELS  # noqa: E402

MVTEC_CATEGORIES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper',
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

VL_MODEL_TYPES = {
    'WinClipDetector', 'AnomalyCLIPDetector', 'AnomalyCLIPOfficialDetector',
    'AnoVLDetector', 'MuScDetector', 'AdaCLIPDetector', 'AACLIPDetector',
    'AnomalyDINODetector',
}


def find_config(method):
    config_dir = os.path.join(ROOT, 'configs', method)
    if not os.path.isdir(config_dir):
        return None
    for pat in ['*mvtec_strict*.py', '*mvtec*.py']:
        matches = [m for m in glob.glob(os.path.join(config_dir, pat))
                   if '__pycache__' not in m]
        if matches:
            return matches[0]
    return None


def get_img_size(cfg):
    if hasattr(cfg, 'img_size'):
        return cfg.img_size
    for key in ['test_pipeline', 'train_pipeline']:
        for step in getattr(cfg, key, []):
            if isinstance(step, dict) and step.get('type') == 'ResizeAD':
                return step.get('size', 256)
    return 256


def collect_test_images(data_root, img_size, use_clip_norm):
    mean, std = (CLIP_MEAN, CLIP_STD) if use_clip_norm else (IMAGENET_MEAN, IMAGENET_STD)
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    images = []
    for cat in MVTEC_CATEGORIES:
        test_dir = os.path.join(data_root, cat, 'test')
        if not os.path.isdir(test_dir):
            continue
        # Normal
        p = os.path.join(test_dir, 'good', '000.png')
        if os.path.exists(p):
            images.append(tfm(Image.open(p).convert('RGB')))
        # Anomalous — first defect
        defects = sorted(d for d in os.listdir(test_dir)
                         if d != 'good' and os.path.isdir(os.path.join(test_dir, d)))
        if defects:
            p = os.path.join(test_dir, defects[0], '000.png')
            if os.path.exists(p):
                images.append(tfm(Image.open(p).convert('RGB')))
    return images


def _try_forward(model, inp):
    """Try predict, then tensor, then bare forward. Return (result, mode_str)."""
    for mode_name, call in [
        ('predict', lambda: model.forward(inp, mode='predict')),
        ('tensor', lambda: model.forward(inp, mode='tensor')),
        ('bare', lambda: model.forward(inp)),
    ]:
        try:
            return call(), mode_name
        except Exception:
            continue
    raise RuntimeError('All forward modes failed')


def _setup_dummy_banks(model, device):
    """Best-effort setup of dummy internal state for memory-bank models."""
    dummy = torch.randn(1, 3, 256, 256).to(device)

    # PatchCore / MemoryBankHead — set numpy memory_bank
    head = getattr(model, 'head', None)
    if head is not None:
        if hasattr(head, 'memory_bank'):
            feat_dim = 512
            try:
                with torch.no_grad():
                    feats = model.extract_feat(dummy)
                f = feats[0] if isinstance(feats, (list, tuple)) else feats
                feat_dim = f.reshape(f.shape[0], -1).shape[-1]
            except Exception:
                pass
            head.memory_bank = np.random.randn(10000, feat_dim).astype(np.float32)

        # memory-bank head — set per-layer feature stats
        if hasattr(head, 'mean_list'):
            try:
                with torch.no_grad():
                    feats = model.extract_feat(dummy)
                head.mean_list = [f.mean(dim=[0, 2, 3]) for f in feats]
                head.std_list = [f.std(dim=[0, 2, 3]) + 1e-6 for f in feats]
                # Also build per-layer memory if needed
                if hasattr(head, 'features'):
                    head.features = {i: f.flatten(2).permute(2, 0, 1)
                                     for i, f in enumerate(feats)}
            except Exception:
                pass

        # PaDiM head — set Gaussian stats
        if hasattr(head, 'embedding_ids'):
            try:
                with torch.no_grad():
                    feats = model.extract_feat(dummy)
                if hasattr(head, 'mean_list'):
                    head.mean_list = [f.mean(dim=0) for f in feats]
                if hasattr(head, 'cov_list'):
                    head.cov_list = [torch.eye(f.shape[0]) for f in feats]
                if hasattr(head, '_fitted'):
                    head._fitted = True
            except Exception:
                pass

        # DFKDE / DFM — fit with dummy data
        if hasattr(head, '_fitted'):
            head._fitted = True
        if hasattr(head, 'pca') and hasattr(head.pca, 'fit'):
            try:
                with torch.no_grad():
                    feats = model.extract_feat(dummy)
                f = feats[0] if isinstance(feats, (list, tuple)) else feats
                dummy_data = f.flatten(1).T.cpu().numpy()
                head.pca.fit(dummy_data)
            except Exception:
                pass

    # memory-bank — set memory bank tensor directly
    if hasattr(model, 'memory_bank'):
        try:
            with torch.no_grad():
                feats = model.extract_feat(dummy)
            f = feats[0] if isinstance(feats, (list, tuple)) else feats
            feat_dim = f.shape[1]
            model.memory_bank = torch.randn(1000, feat_dim).to(device)
        except Exception:
            pass

    # RegAD — set support bank
    if hasattr(model, 'support_bank'):
        try:
            with torch.no_grad():
                feats = model.extract_feat(dummy)
            f = feats[0] if isinstance(feats, (list, tuple)) else feats
            model.support_bank = f.flatten(1).cpu()
            if hasattr(model, '_fitted'):
                model._fitted = True
        except Exception:
            pass

    # AnomalyDINO / VL models with memory
    if hasattr(model, 'memory_bank') and not isinstance(model.memory_bank, torch.Tensor):
        try:
            with torch.no_grad():
                feats = model.extract_feat(dummy)
            model.memory_bank = feats[0]
        except Exception:
            pass


def benchmark_one(method, data_root, device, warmup=10, runs=100):
    cfg_path = find_config(method)
    if cfg_path is None:
        print(f"[SKIP] {method}: no config found")
        return None

    print(f"\n{'='*60}")
    print(f"[BENCH] {method}  ({os.path.basename(cfg_path)})")

    cfg = Config.fromfile(cfg_path)
    apply_data_root_overrides(cfg)
    img_size = get_img_size(cfg)
    model_type = cfg.get('model', {}).get('type', '')
    is_vl = model_type in VL_MODEL_TYPES
    print(f"  img_size={img_size}  type={model_type}  vl={is_vl}")

    # Build model
    try:
        model = MODELS.build(cfg.model).to(device).eval()
    except Exception as e:
        print(f"  [FAIL] build: {e}")
        traceback.print_exc()
        return None

    # Memory bank setup
    try:
        _setup_dummy_banks(model, device)
    except Exception:
        pass

    # Load images
    images = collect_test_images(data_root, img_size, is_vl)
    if not images:
        print(f"  [SKIP] no images at {data_root}")
        return None
    print(f"  {len(images)} images loaded")

    # Detect which forward mode works (prefer predict)
    sample_inp = images[0].unsqueeze(0).to(device)
    with torch.no_grad():
        _, best_mode = _try_forward(model, sample_inp)
    if best_mode != 'predict':
        print(f"  NOTE: using mode='{best_mode}' (predict unavailable)")
    forward_mode = best_mode

    # ---- Warmup ----
    with torch.no_grad():
        for img in images:
            inp = img.unsqueeze(0).to(device)
            for _ in range(warmup):
                _try_forward(model, inp)

    # ---- Timed runs ----
    per_image_ms = []
    with torch.no_grad():
        for img in images:
            inp = img.unsqueeze(0).to(device)
            times = []
            for _ in range(runs):
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                _try_forward(model, inp)
                torch.cuda.synchronize(device)
                t1 = time.perf_counter()
                times.append(t1 - t0)
            per_image_ms.append(float(np.mean(times)) * 1000)

    avg_ms = float(np.mean(per_image_ms))
    fps = 1000.0 / avg_ms
    std_ms = float(np.std(per_image_ms))
    result = {
        'method': method,
        'img_size': img_size,
        'avg_ms_per_img': round(avg_ms, 2),
        'std_ms': round(std_ms, 2),
        'fps': round(fps, 1),
        'n_images': len(images),
        'warmup': warmup,
        'runs': runs,
        'forward_mode': forward_mode,
    }
    print(f"  >> {avg_ms:.2f} ± {std_ms:.2f} ms/img  |  {fps:.1f} FPS  [{forward_mode}]")
    return result


def _benchmark_methods(methods, data_root, device, warmup, runs):
    """Run each method while the caller owns checkpoint policy scope."""
    results = []
    for method in methods:
        try:
            result = benchmark_one(method, data_root, device, warmup, runs)
            if result:
                results.append(result)
        except Exception as exc:
            print(f"  [ERROR] {method}: {exc}")
            traceback.print_exc()
    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark BaoIAD inference speed')
    parser.add_argument('--methods', required=True, help='Comma-separated method names')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--data-root', default='data/mvtec_ad')
    parser.add_argument('--output', required=True)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--runs', type=int, default=100)
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
    args = parser.parse_args()

    from baoiad.runtime import configure_offline_mode

    configure_offline_mode(args.offline)

    device = torch.device(f'cuda:{args.gpu}')
    methods = [m.strip() for m in args.methods.split(',') if m.strip()]

    from baoiad.checkpoint import checkpoint_loading_policy

    with checkpoint_loading_policy(args.trusted_checkpoint):
        results = _benchmark_methods(
            methods, args.data_root, device, args.warmup, args.runs
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*75}")
    print(f"{'Method':<22} {'Size':>4} {'ms/img':>8} {'±std':>6} {'FPS':>8}  {'Mode':>8}")
    print('-' * 75)
    for r in sorted(results, key=lambda x: x['fps'], reverse=True):
        mode = r.get('forward_mode', '?')
        print(f"{r['method']:<22} {r['img_size']:>4} {r['avg_ms_per_img']:>8.2f} "
              f"{r['std_ms']:>6.2f} {r['fps']:>8.1f}  {mode:>8}")
    print('=' * 75)
    print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()

"""Helpers for running the official ResAD protocol inside BaoIAD."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import os
import shutil
import sys
import zipfile
from argparse import Namespace
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from baoiad.utils.freia import patch_freia_soft_permutation_rvs

_RESAD_DATASET_CLASS_NAMES = {
    'mvtec': [
        'bottle',
        'cable',
        'capsule',
        'carpet',
        'grid',
        'hazelnut',
        'leather',
        'metal_nut',
        'pill',
        'screw',
        'tile',
        'toothbrush',
        'transistor',
        'wood',
        'zipper',
    ],
}

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def patch_numpy_sctypes() -> None:
    """Restore ``np.sctypes`` for legacy imgaug-based official code."""
    if hasattr(np, 'sctypes'):
        return

    np.sctypes = {  # type: ignore[attr-defined]
        'int': [np.int8, np.int16, np.int32, np.int64],
        'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
        'float': [np.float16, np.float32, np.float64],
        'complex': [np.complex64, np.complex128],
        'others': [np.bool_, np.object_, np.str_, np.bytes_],
    }


def patch_timm_legacy_imports() -> None:
    """Alias legacy timm import paths used by upstream ResAD."""
    try:
        import timm.layers.create_act as create_act_mod
        import timm.layers.create_attn as create_attn_mod
        import timm.layers.helpers as helpers_mod
    except ImportError:  # timm<0.9 compatibility
        import timm.models.layers.create_act as create_act_mod
        import timm.models.layers.create_attn as create_attn_mod
        import timm.models.layers.helpers as helpers_mod

    sys.modules['timm.models.layers.create_act'] = create_act_mod
    sys.modules['timm.models.layers.create_attn'] = create_attn_mod
    sys.modules['timm.models.layers.helpers'] = helpers_mod
    sys.modules['timm.layers.create_act'] = create_act_mod
    sys.modules['timm.layers.create_attn'] = create_attn_mod
    sys.modules['timm.layers.helpers'] = helpers_mod


def _resolve_timm_cached_checkpoint(timm_module, model_name: str) -> str:
    """Resolve a local checkpoint path for official timm backbones."""
    fallback_filenames = {
        'wide_resnet50_2': 'wide_resnet50_2-95faca4d.pth',
    }

    filename = fallback_filenames.get(model_name, '')
    if not filename:
        get_pretrained_cfg = getattr(timm_module, 'get_pretrained_cfg', None)
        if get_pretrained_cfg is not None:
            try:
                pretrained_cfg = get_pretrained_cfg(model_name)
            except Exception:
                pretrained_cfg = None
            if pretrained_cfg is not None:
                if isinstance(pretrained_cfg, dict):
                    pretrained_url = pretrained_cfg.get('url', '')
                else:
                    pretrained_url = getattr(pretrained_cfg, 'url', '')
                if pretrained_url:
                    filename = os.path.basename(pretrained_url)

    if not filename:
        return ''

    candidate = os.path.join(torch.hub.get_dir(), 'checkpoints', filename)
    return candidate if os.path.exists(candidate) else ''


def patch_timm_local_pretrained() -> None:
    """Patch ``timm.create_model`` to reuse cached local checkpoints."""
    import timm

    current_create_model = timm.create_model
    original_create_model = getattr(current_create_model, '_baoiad_original_create_model', current_create_model)

    def _patched_create_model(model_name, *args, pretrained=False, **kwargs):
        checkpoint_path = ''
        if pretrained:
            checkpoint_path = _resolve_timm_cached_checkpoint(timm, model_name)

        model = original_create_model(model_name, *args, pretrained=bool(pretrained and not checkpoint_path), **kwargs)
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
            if isinstance(state_dict, dict) and state_dict:
                first_key = next(iter(state_dict))
                if first_key.startswith('module.'):
                    state_dict = {k[7:]: v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
        return model

    _patched_create_model._baoiad_original_create_model = original_create_model  # type: ignore[attr-defined]
    timm.create_model = _patched_create_model


def resolve_official_reference_dir(
    root: str | Path,
    class_name: str,
    *,
    mvtec_class_names: list[str],
    visa_class_names: list[str],
) -> Path | None:
    """Resolve the directory used for official train-time few-shot sampling."""
    root = Path(root)
    if class_name in mvtec_class_names:
        candidate = root / class_name / 'train' / 'good'
        if candidate.is_dir():
            return candidate
        raise FileNotFoundError(f'MVTec normal reference directory not found: {candidate}')

    if class_name in visa_class_names:
        candidates = [
            root / class_name / 'Data' / 'Images' / 'Normal',
            root / class_name / 'train' / 'good',
        ]
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
        raise FileNotFoundError(
            'VisA normal reference directory not found. Tried: '
            + ', '.join(str(candidate) for candidate in candidates)
        )

    return None


def patch_official_reference_sampling() -> None:
    """Patch the official helper to support BaoIAD-style VisA layout."""
    import utils as official_utils
    from datasets.mvtec import MVTEC
    from datasets.visa import VISA

    current_fn = official_utils.get_random_normal_images
    original_fn = getattr(current_fn, '_baoiad_original_get_random_normal_images', current_fn)

    def _patched(root, class_name, num_shot=4):
        root_dir = resolve_official_reference_dir(
            root,
            class_name,
            mvtec_class_names=MVTEC.CLASS_NAMES,
            visa_class_names=VISA.CLASS_NAMES,
        )
        if root_dir is None:
            return original_fn(root, class_name, num_shot=num_shot)

        filenames = [name for name in os.listdir(root_dir) if (root_dir / name).is_file()]
        if not filenames:
            raise FileNotFoundError(f'No reference images found in {root_dir}')
        indices = np.random.randint(len(filenames), size=num_shot).tolist()
        return [str(root_dir / filenames[index]) for index in indices]

    _patched._baoiad_original_get_random_normal_images = original_fn  # type: ignore[attr-defined]
    official_utils.get_random_normal_images = _patched


def prepare_official_resad_import(ref_repo_root: str | Path, device: str | None = None) -> Path:
    """Apply compatibility patches and add the official repo to ``sys.path``."""
    patch_numpy_sctypes()
    patch_timm_legacy_imports()
    patch_timm_local_pretrained()
    patch_freia_soft_permutation_rvs(device=device)

    ref_repo_root = Path(ref_repo_root).resolve()
    ref_repo_root_str = str(ref_repo_root)
    if ref_repo_root_str not in sys.path:
        sys.path.insert(0, ref_repo_root_str)
    return ref_repo_root


def load_official_main(ref_repo_root: str | Path, device: str | None = None):
    """Load the official ResAD ``main.py`` module."""
    ref_repo_root = prepare_official_resad_import(ref_repo_root, device=device)
    module_path = ref_repo_root / 'main.py'

    spec = importlib.util.spec_from_file_location('resad_official_main', module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    patch_official_reference_sampling()
    return module


def initialize_official_seeds(module, seed: int = 42) -> None:
    """Run the upstream ResAD seed init path before training/evaluation."""
    init_fn = getattr(module, 'init_seeds', None)
    if callable(init_fn):
        init_fn(seed)
        return

    utils_module = sys.modules.get('utils')
    if utils_module is not None:
        fallback = getattr(utils_module, 'init_seeds', None)
        if callable(fallback):
            fallback(seed)


def patch_official_dataloader_num_workers(module, num_workers: int | None = None):
    """Optionally override upstream ResAD DataLoader worker count at runtime."""
    if num_workers is None:
        return module

    original_dataloader = getattr(module.DataLoader, '_baoiad_original_dataloader', module.DataLoader)

    @wraps(original_dataloader)
    def _patched_dataloader(*args, **kwargs):
        kwargs['num_workers'] = int(num_workers)
        return original_dataloader(*args, **kwargs)

    _patched_dataloader._baoiad_original_dataloader = original_dataloader  # type: ignore[attr-defined]
    module.DataLoader = _patched_dataloader
    return module


def _resolve_path(path: str | Path, repo_root: str | Path) -> str:
    path = Path(path)
    if path.is_absolute():
        return str(path)
    return str((Path(repo_root) / path).resolve())


def get_setting_test_dataset(setting: str) -> str | None:
    """Infer the held-out dataset name from ``<seen>_to_<unseen>`` settings."""
    if '_to_' not in setting:
        return None
    return setting.split('_to_', maxsplit=1)[1]


def get_reference_feature_class_names(dataset: str) -> list[str]:
    """Return expected class names for pre-extracted official reference features."""
    if dataset not in _RESAD_DATASET_CLASS_NAMES:
        raise ValueError(
            f'Unsupported ResAD reference-feature dataset "{dataset}". '
            f'Known datasets: {sorted(_RESAD_DATASET_CLASS_NAMES)}'
        )
    return list(_RESAD_DATASET_CLASS_NAMES[dataset])


def _is_raw_few_shot_dir(root_dir: Path) -> bool:
    for class_dir in sorted(path for path in root_dir.iterdir() if path.is_dir()):
        if (class_dir / 'train' / 'good').is_dir():
            return True
    return False


def validate_reference_feature_dir(
    root_dir: str | Path,
    *,
    class_names: Sequence[str],
    feature_levels: int = 3,
) -> Path:
    """Validate the official ``layer{1,2,3}.npy`` directory structure."""
    root_dir = Path(root_dir).resolve()
    if not root_dir.exists():
        raise FileNotFoundError(f'Reference feature directory not found: {root_dir}')
    if not root_dir.is_dir():
        raise NotADirectoryError(f'Reference feature path is not a directory: {root_dir}')

    if _is_raw_few_shot_dir(root_dir):
        raise FileNotFoundError(
            'Official ResAD expected extracted reference features, but '
            f'"{root_dir}" looks like a raw few-shot image directory '
            '(contains "<class>/train/good").'
        )

    required_filenames = tuple(f'layer{level}.npy' for level in range(1, feature_levels + 1))
    missing_paths: list[str] = []
    for class_name in class_names:
        class_dir = root_dir / class_name
        if not class_dir.is_dir():
            missing_paths.append(str(class_dir))
            continue
        for filename in required_filenames:
            feature_path = class_dir / filename
            if not feature_path.is_file():
                missing_paths.append(str(feature_path))

    if missing_paths:
        preview = ', '.join(missing_paths[:6])
        if len(missing_paths) > 6:
            preview = f'{preview}, ...'
        raise FileNotFoundError(
            f'Incomplete ResAD reference features under "{root_dir}". Missing: {preview}'
        )
    return root_dir


def build_reference_feature_manifest(
    root_dir: str | Path,
    *,
    class_names: Sequence[str],
    feature_levels: int = 3,
) -> dict[str, Any]:
    """Build a stable manifest for one official ResAD ref-feature directory."""
    root_dir = validate_reference_feature_dir(
        root_dir,
        class_names=class_names,
        feature_levels=feature_levels,
    )
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0

    for class_name in class_names:
        class_dir = root_dir / class_name
        for level in range(1, feature_levels + 1):
            feature_path = class_dir / f'layer{level}.npy'
            relative_path = feature_path.relative_to(root_dir).as_posix()
            stat = feature_path.stat()
            digest.update(relative_path.encode('utf-8'))
            digest.update(f':{stat.st_size}:'.encode('utf-8'))
            with open(feature_path, 'rb') as fopen:
                for chunk in iter(lambda: fopen.read(1024 * 1024), b''):
                    digest.update(chunk)
            file_count += 1
            total_bytes += stat.st_size

    return dict(
        root_dir=str(root_dir),
        class_names=list(class_names),
        feature_levels=feature_levels,
        file_count=file_count,
        total_bytes=total_bytes,
        sha256=digest.hexdigest(),
    )


def parse_resad_official_metrics(text: str) -> dict[str, float]:
    """Parse the merged-average metrics that the official ResAD script prints."""
    import re

    metrics: dict[str, float] = {}
    scalar_patterns = {
        'image_auroc': [
            r'image AUROC\s*=\s*([0-9]*\.?[0-9]+)',
        ],
        'pixel_auroc': [
            r'pixel AUROC\s*=\s*([0-9]*\.?[0-9]+)',
        ],
        'image_ap': [
            r'image AP\s*=\s*([0-9]*\.?[0-9]+)',
        ],
        'pixel_ap': [
            r'pixel AP\s*=\s*([0-9]*\.?[0-9]+)',
        ],
        'train_loss': [
            r'train_loss\s*=\s*([0-9]*\.?[0-9]+)',
            r'train_loss:\s*([0-9]*\.?[0-9]+)',
        ],
    }

    for name, patterns in scalar_patterns.items():
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                metrics[name] = float(matches[-1])
                break

    merged_average_matches = re.findall(
        (
            r'\(Merged\)\s+Average Image AUC \| AP \| F1_Score:\s*'
            r'([0-9]*\.?[0-9]+)\s*\|\s*([0-9]*\.?[0-9]+)\s*\|\s*([0-9]*\.?[0-9]+),\s*'
            r'Average Pixel AUC \| AP \| F1_Score \| AUPRO:\s*'
            r'([0-9]*\.?[0-9]+)\s*\|\s*([0-9]*\.?[0-9]+)\s*\|\s*([0-9]*\.?[0-9]+)\s*\|\s*'
            r'(-?[0-9]*\.?[0-9]+)'
        ),
        text,
    )
    if merged_average_matches:
        last = merged_average_matches[-1]
        metrics.update(
            image_auroc=float(last[0]),
            image_ap=float(last[1]),
            image_f1=float(last[2]),
            pixel_auroc=float(last[3]),
            pixel_ap=float(last[4]),
            pixel_f1=float(last[5]),
            aupro=float(last[6]),
        )
    return metrics


def ensure_resad_few_shot_dir(
    few_shot_dir: str | Path,
    *,
    archive_path: str | Path | None = None,
    dataset: str = 'mvtec',
) -> Path:
    """Ensure official few-shot images exist, extracting them from ``ResAD-data.zip`` if needed."""
    few_shot_dir = Path(few_shot_dir).resolve()
    if few_shot_dir.is_dir():
        return few_shot_dir

    if archive_path is None:
        raise FileNotFoundError(
            f'ResAD few-shot directory not found: {few_shot_dir}. '
            'No archive path was provided for extraction.'
        )

    archive_path = Path(archive_path).resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(
            f'ResAD few-shot directory not found: {few_shot_dir}. '
            f'Archive also missing: {archive_path}'
        )

    prefix = f'data/4shot/{dataset}/'
    extracted = 0
    with zipfile.ZipFile(archive_path, 'r') as zf:
        for member in zf.namelist():
            if not member.startswith(prefix) or member.endswith('/'):
                continue
            relative = Path(member[len(prefix):])
            if not relative.parts:
                continue
            target = few_shot_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target, 'wb') as dst:
                shutil.copyfileobj(src, dst)
            extracted += 1

    if extracted == 0:
        raise FileNotFoundError(
            f'Archive "{archive_path}" does not contain official few-shot data under "{prefix}".'
        )
    return few_shot_dir


class _FewShotImageDataset(Dataset):
    """Minimal dataset matching the official reference-feature extraction pipeline."""

    def __init__(self, root_dir: str | Path, image_size: int = 224) -> None:
        self.root_dir = Path(root_dir).resolve()
        if not self.root_dir.is_dir():
            raise FileNotFoundError(f'Few-shot image directory not found: {self.root_dir}')

        valid_suffixes = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.JPG', '.PNG'}
        self.image_paths = sorted(
            path for path in self.root_dir.iterdir()
            if path.is_file() and path.suffix in valid_suffixes
        )
        if not self.image_paths:
            raise FileNotFoundError(f'No few-shot images found in {self.root_dir}')

        self.transform = T.Compose([
            T.Resize(image_size, T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.open(self.image_paths[index]).convert('RGB')
        return self.transform(image)


class _ReferenceImageDataset(Dataset):
    """Dataset over explicit image paths for cached official train refs."""

    def __init__(self, image_paths: Sequence[Path], transform) -> None:
        self.image_paths = list(image_paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image_path = self.image_paths[index]
        with open(image_path, 'rb') as fopen:
            image = Image.open(fopen).convert('RGB')
        return self.transform(image)


def prepare_reference_features(
    *,
    ref_repo_root: str | Path,
    save_dir: str | Path,
    dataset: str = 'mvtec',
    few_shot_dir: str | Path | None = None,
    archive_path: str | Path | None = None,
    device: str = 'cuda:0',
    batch_size: int = 8,
    num_workers: int = 8,
    force: bool = False,
) -> Path:
    """Generate official reference features if they are missing or incomplete."""
    class_names = get_reference_feature_class_names(dataset)
    save_dir = Path(save_dir).resolve()
    if not force:
        try:
            return validate_reference_feature_dir(save_dir, class_names=class_names)
        except FileNotFoundError:
            pass

    if few_shot_dir is None:
        raise FileNotFoundError('Few-shot image directory must be provided to prepare reference features.')

    few_shot_dir = ensure_resad_few_shot_dir(few_shot_dir, archive_path=archive_path, dataset=dataset)
    prepare_official_resad_import(ref_repo_root, device=device)
    import timm

    encoder = timm.create_model(
        'wide_resnet50_2',
        features_only=True,
        out_indices=(1, 2, 3),
        pretrained=True,
    ).eval()
    encoder = encoder.to(device)

    save_dir.mkdir(parents=True, exist_ok=True)
    for class_name in class_names:
        class_root = few_shot_dir / class_name / 'train' / 'good'
        dataset_obj = _FewShotImageDataset(class_root)
        loader = DataLoader(
            dataset_obj,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )

        layer_batches = [[], [], []]
        for images in loader:
            with torch.no_grad():
                features = encoder(images.to(device))
            for level, feature in enumerate(features):
                layer_batches[level].append(feature.detach().cpu())

        class_dir = save_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for level, batches in enumerate(layer_batches, start=1):
            features = torch.cat(batches, dim=0)
            channels = features.shape[1]
            flattened = features.permute(0, 2, 3, 1).reshape(-1, channels).numpy()
            np.save(class_dir / f'layer{level}.npy', flattened)

    validate_reference_feature_dir(save_dir, class_names=class_names)
    return save_dir


class _ImageTensorCache:
    """Cache preprocessed official train-time normal images on CPU."""

    def __init__(
        self,
        root: str | Path,
        *,
        mvtec_class_names: Sequence[str],
        visa_class_names: Sequence[str],
        image_size: int = 224,
        batch_size: int = 64,
        num_workers: int = 4,
    ) -> None:
        self.root = Path(root).resolve()
        self.mvtec_class_names = list(mvtec_class_names)
        self.visa_class_names = list(visa_class_names)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = T.Compose([
            T.Resize(image_size, T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])
        self._class_tensors: dict[str, torch.Tensor] = {}

    def _resolve_class_dir(self, class_name: str) -> Path:
        root_dir = resolve_official_reference_dir(
            self.root,
            class_name,
            mvtec_class_names=self.mvtec_class_names,
            visa_class_names=self.visa_class_names,
        )
        if root_dir is None:
            raise FileNotFoundError(
                f'Unable to resolve official normal reference directory for class "{class_name}" under {self.root}'
            )
        return root_dir

    def _load_class_tensor(self, class_name: str) -> torch.Tensor:
        class_dir = self._resolve_class_dir(class_name)
        valid_suffixes = {'.png', '.jpg', '.jpeg', '.bmp', '.JPG', '.PNG'}
        filenames = [name for name in os.listdir(class_dir) if Path(name).suffix in valid_suffixes]
        if not filenames:
            raise FileNotFoundError(f'No normal images found in {class_dir}')

        image_paths = [class_dir / filename for filename in filenames]
        dataset = _ReferenceImageDataset(image_paths, self.transform)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
        )
        batches = [batch.contiguous() for batch in loader]
        tensor = torch.cat(batches, dim=0).contiguous()
        self._class_tensors[class_name] = tensor
        return tensor

    def ensure_class(self, class_name: str) -> torch.Tensor:
        tensor = self._class_tensors.get(class_name)
        if tensor is not None:
            return tensor
        return self._load_class_tensor(class_name)

    def preload_classes(self, class_names: Sequence[str]) -> None:
        for class_name in class_names:
            self.ensure_class(class_name)

    def sample(self, class_name: str, num_shot: int, device: str | torch.device) -> torch.Tensor:
        tensor = self.ensure_class(class_name)
        indices = np.random.randint(tensor.shape[0], size=num_shot).tolist()
        return tensor[indices].to(device)


def patch_official_train_reference_tensor_cache(
    module,
    *,
    setting: str,
    train_dataset_dir: str | Path,
    preload: bool = False,
) -> SimpleNamespace:
    """Patch upstream train-time reference sampling to reuse cached tensors."""
    import utils as official_utils
    from datasets.mvtec import MVTEC
    from datasets.visa import VISA

    cache = _ImageTensorCache(
        train_dataset_dir,
        mvtec_class_names=MVTEC.CLASS_NAMES,
        visa_class_names=VISA.CLASS_NAMES,
    )
    if preload:
        seen_classes = module.SETTINGS[setting]['seen']
        cache.preload_classes(seen_classes)

    current_fn = module.get_mc_reference_features
    original_fn = getattr(current_fn, '_baoiad_original_get_mc_reference_features', current_fn)

    def _patched_get_mc_reference_features(encoder, root, class_names, device, num_shot=4):
        reference_features = {}
        for class_name in np.unique(class_names):
            images = cache.sample(class_name, num_shot, device)
            with torch.no_grad():
                features = encoder(images)
                for level in range(len(features)):
                    bs, channels, _, _ = features[level].shape
                    features[level] = features[level].permute(0, 2, 3, 1).reshape(-1, channels)
                reference_features[class_name] = features
        return reference_features

    _patched_get_mc_reference_features._baoiad_original_get_mc_reference_features = original_fn  # type: ignore[attr-defined]
    official_utils.get_mc_reference_features = _patched_get_mc_reference_features
    module.get_mc_reference_features = _patched_get_mc_reference_features
    return SimpleNamespace(cache=cache, original_fn=original_fn)


def build_official_namespace(cfg: Mapping[str, Any], repo_root: str | Path) -> Namespace:
    """Build the official ResAD ``argparse.Namespace`` from config values."""
    repo_root = Path(repo_root).resolve()

    work_dir = _resolve_path(cfg.get('work_dir', 'runs/resad_official_visa_to_mvtec'), repo_root)
    checkpoint_path = _resolve_path(
        cfg.get('checkpoint_path', str(Path(work_dir) / 'checkpoints')),
        repo_root,
    )

    return Namespace(
        ref_repo_root=_resolve_path(cfg.get('ref_repo_root', '.refs/ResAD'), repo_root),
        setting=cfg.get('setting', 'visa_to_mvtec'),
        train_dataset_dir=_resolve_path(cfg.get('train_dataset_dir', 'data/VisA_20220922'), repo_root),
        test_dataset_dir=_resolve_path(cfg.get('test_dataset_dir', 'data/mvtec_ad'), repo_root),
        test_ref_feature_dir=_resolve_path(
            cfg.get('test_ref_feature_dir', 'pretrained/resad/ref_features/w50/mvtec_4shot'),
            repo_root,
        ),
        batch_size=int(cfg.get('batch_size', 32)),
        lr=float(cfg.get('lr', 1e-5)),
        epochs=int(cfg.get('epochs', 100)),
        device=str(cfg.get('device', 'cuda:0')),
        checkpoint_path=checkpoint_path,
        eval_freq=int(cfg.get('eval_freq', 1)),
        backbone=str(cfg.get('backbone', 'wide_resnet50_2')),
        flow_arch=str(cfg.get('flow_arch', 'conditional_flow_model')),
        feature_levels=int(cfg.get('feature_levels', 3)),
        coupling_layers=int(cfg.get('coupling_layers', 10)),
        clamp_alpha=float(cfg.get('clamp_alpha', 1.9)),
        pos_embed_dim=int(cfg.get('pos_embed_dim', 256)),
        pos_beta=float(cfg.get('pos_beta', 0.05)),
        margin_tau=float(cfg.get('margin_tau', 0.1)),
        bgspp_lambda=float(cfg.get('bgspp_lambda', 1.0)),
        fdm_alpha=float(cfg.get('fdm_alpha', 0.4)),
        num_embeddings=int(cfg.get('num_embeddings', 1536)),
        train_ref_shot=int(cfg.get('train_ref_shot', 4)),
        num_ref_shot=int(cfg.get('num_ref_shot', 4)),
        num_workers=(
            None if cfg.get('num_workers', None) is None else int(cfg.get('num_workers'))
        ),
        cache_train_ref_tensors=bool(cfg.get('cache_train_ref_tensors', False)),
        cache_train_ref_preload=bool(cfg.get('cache_train_ref_preload', False)),
        work_dir=work_dir,
    )


def validate_official_inputs(args: Namespace) -> None:
    """Validate required filesystem inputs for the official protocol."""
    required_dirs = {
        'ref_repo_root': args.ref_repo_root,
        'train_dataset_dir': args.train_dataset_dir,
        'test_dataset_dir': args.test_dataset_dir,
        'test_ref_feature_dir': args.test_ref_feature_dir,
    }
    for name, path in required_dirs.items():
        if not Path(path).exists():
            raise FileNotFoundError(f'ResAD official protocol path not found: {name}={path}')

    test_dataset = get_setting_test_dataset(args.setting)
    if test_dataset in _RESAD_DATASET_CLASS_NAMES:
        validate_reference_feature_dir(
            args.test_ref_feature_dir,
            class_names=get_reference_feature_class_names(test_dataset),
            feature_levels=3,
        )


def ensure_visa_split_csv(visa_root: str | Path) -> Path:
    """Create the official VisA ``split_csv/1cls.csv`` if it is missing."""
    visa_root = Path(visa_root).resolve()
    csv_path = visa_root / 'split_csv' / '1cls.csv'
    if csv_path.exists():
        return csv_path

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    valid_suffixes = {'.png', '.jpg', '.jpeg', '.bmp', '.JPG', '.PNG'}

    for class_dir in sorted(path for path in visa_root.iterdir() if path.is_dir()):
        class_name = class_dir.name

        train_good = class_dir / 'train' / 'good'
        for image_path in sorted(train_good.iterdir()) if train_good.exists() else []:
            if image_path.suffix not in valid_suffixes:
                continue
            rows.append(dict(
                object=class_name,
                split='train',
                label='normal',
                image=str(image_path.relative_to(visa_root)),
                mask='',
            ))

        test_root = class_dir / 'test'
        if not test_root.exists():
            continue
        for defect_dir in sorted(path for path in test_root.iterdir() if path.is_dir()):
            is_good = defect_dir.name == 'good'
            gt_dir = class_dir / 'ground_truth' / defect_dir.name
            for image_path in sorted(defect_dir.iterdir()):
                if image_path.suffix not in valid_suffixes:
                    continue

                mask_rel = ''
                if not is_good:
                    stem = image_path.stem
                    candidates = [
                        gt_dir / f'{stem}.png',
                        gt_dir / f'{stem}_mask.png',
                        gt_dir / f'{stem}.bmp',
                        gt_dir / f'{stem}.jpg',
                    ]
                    for candidate in candidates:
                        if candidate.exists():
                            mask_rel = str(candidate.relative_to(visa_root))
                            break

                rows.append(dict(
                    object=class_name,
                    split='test',
                    label='normal' if is_good else 'anomaly',
                    image=str(image_path.relative_to(visa_root)),
                    mask=mask_rel,
                ))

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['object', 'split', 'label', 'image', 'mask'])
        writer.writeheader()
        writer.writerows(rows)
    return csv_path

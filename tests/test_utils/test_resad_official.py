"""Tests for official ResAD protocol helpers."""

import csv
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

import baoiad.utils.freia as freia_utils
from baoiad.utils.compat import ensure_legacy_imgaug_compat
from baoiad.utils.resad_official import (
    build_reference_feature_manifest,
    build_official_namespace,
    ensure_resad_few_shot_dir,
    ensure_visa_split_csv,
    initialize_official_seeds,
    parse_resad_official_metrics,
    patch_numpy_sctypes,
    patch_official_dataloader_num_workers,
    patch_official_train_reference_tensor_cache,
    patch_timm_legacy_imports,
    patch_timm_local_pretrained,
    resolve_official_reference_dir,
    validate_reference_feature_dir,
    validate_official_inputs,
)

ROOT = Path(__file__).resolve().parents[2]


def _write_fake_rgb(path: Path, size=(64, 64)):
    array = np.random.randint(0, 255, size + (3,), dtype=np.uint8)
    Image.fromarray(array).save(path)


def test_patch_numpy_sctypes_restores_legacy_attribute(monkeypatch):
    monkeypatch.delattr(np, 'sctypes', raising=False)
    patch_numpy_sctypes()
    assert hasattr(np, 'sctypes')
    assert np.float32 in np.sctypes['float']


def test_ensure_legacy_imgaug_compat_restores_numpy_and_collections(monkeypatch):
    import collections

    monkeypatch.delattr(np, 'sctypes', raising=False)
    monkeypatch.delattr(collections, 'Iterable', raising=False)
    monkeypatch.delattr(collections, 'Mapping', raising=False)
    monkeypatch.delattr(collections, 'MutableMapping', raising=False)
    monkeypatch.delattr(collections, 'Sequence', raising=False)

    ensure_legacy_imgaug_compat()

    assert hasattr(np, 'sctypes')
    assert hasattr(collections, 'Iterable')
    assert hasattr(collections, 'Mapping')
    assert hasattr(collections, 'MutableMapping')
    assert hasattr(collections, 'Sequence')


def test_patch_timm_legacy_imports_registers_old_module_names():
    sys.modules.pop('timm.models.layers.create_act', None)
    sys.modules.pop('timm.models.layers.create_attn', None)
    sys.modules.pop('timm.models.layers.helpers', None)

    patch_timm_legacy_imports()

    assert 'timm.models.layers.create_act' in sys.modules
    assert 'timm.models.layers.create_attn' in sys.modules
    assert 'timm.models.layers.helpers' in sys.modules


def test_ensure_visa_split_csv_generates_official_columns(tmp_path):
    visa_root = tmp_path / 'VisA_20220922'
    train_good = visa_root / 'candle' / 'train' / 'good'
    test_good = visa_root / 'candle' / 'test' / 'good'
    test_bad = visa_root / 'candle' / 'test' / 'bad'
    gt_bad = visa_root / 'candle' / 'ground_truth' / 'bad'
    for path in [train_good, test_good, test_bad, gt_bad]:
        path.mkdir(parents=True)

    for path in [
        train_good / '000.JPG',
        test_good / '001.JPG',
        test_bad / '002.JPG',
        gt_bad / '002.png',
    ]:
        path.write_bytes(b'test')

    csv_path = ensure_visa_split_csv(visa_root)
    assert csv_path.exists()

    with open(csv_path, 'r') as f:
        rows = list(csv.DictReader(f))

    assert rows == [
        {
            'object': 'candle',
            'split': 'train',
            'label': 'normal',
            'image': 'candle/train/good/000.JPG',
            'mask': '',
        },
        {
            'object': 'candle',
            'split': 'test',
            'label': 'anomaly',
            'image': 'candle/test/bad/002.JPG',
            'mask': 'candle/ground_truth/bad/002.png',
        },
        {
            'object': 'candle',
            'split': 'test',
            'label': 'normal',
            'image': 'candle/test/good/001.JPG',
            'mask': '',
        },
    ]


def test_build_official_namespace_resolves_repo_relative_paths(tmp_path):
    cfg = {
        'ref_repo_root': '.refs/ResAD',
        'train_dataset_dir': 'data/VisA_20220922',
        'test_dataset_dir': 'data/mvtec_ad',
        'test_ref_feature_dir': 'pretrained/resad/ref_features/w50/mvtec_4shot',
        'work_dir': 'runs/resad_official',
    }

    args = build_official_namespace(cfg, tmp_path)

    assert args.ref_repo_root == str((tmp_path / '.refs/ResAD').resolve())
    assert args.train_dataset_dir == str((tmp_path / 'data/VisA_20220922').resolve())
    assert args.test_dataset_dir == str((tmp_path / 'data/mvtec_ad').resolve())
    assert args.test_ref_feature_dir == str((tmp_path / 'pretrained/resad/ref_features/w50/mvtec_4shot').resolve())
    assert args.checkpoint_path == str((tmp_path / 'runs/resad_official/checkpoints').resolve())
    assert args.num_workers is None


def test_build_official_namespace_parses_optional_num_workers(tmp_path):
    cfg = {
        'num_workers': 0,
        'cache_train_ref_tensors': True,
        'cache_train_ref_preload': True,
    }

    args = build_official_namespace(cfg, tmp_path)

    assert args.num_workers == 0
    assert args.cache_train_ref_tensors is True
    assert args.cache_train_ref_preload is True


def test_resolve_official_reference_dir_supports_baoiad_style_visa(tmp_path):
    visa_root = tmp_path / 'visa'
    (visa_root / 'candle' / 'train' / 'good').mkdir(parents=True)

    resolved = resolve_official_reference_dir(
        visa_root,
        'candle',
        mvtec_class_names=['bottle'],
        visa_class_names=['candle'],
    )

    assert resolved == (visa_root / 'candle' / 'train' / 'good')


def test_validate_reference_feature_dir_rejects_raw_few_shot_layout(tmp_path):
    raw_root = tmp_path / 'mvtec_4shot'
    (raw_root / 'bottle' / 'train' / 'good').mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match='raw few-shot image directory'):
        validate_reference_feature_dir(raw_root, class_names=['bottle'])


def test_validate_official_inputs_requires_complete_reference_features(tmp_path):
    ref_repo_root = tmp_path / '.refs' / 'ResAD'
    ref_repo_root.mkdir(parents=True)
    train_root = tmp_path / 'data' / 'visa'
    train_root.mkdir(parents=True)
    test_root = tmp_path / 'data' / 'mvtec'
    test_root.mkdir(parents=True)
    ref_root = tmp_path / 'pretrained' / 'resad' / 'ref_features' / 'w50' / 'mvtec_4shot'
    (ref_root / 'bottle').mkdir(parents=True)

    args = SimpleNamespace(
        ref_repo_root=str(ref_repo_root),
        train_dataset_dir=str(train_root),
        test_dataset_dir=str(test_root),
        test_ref_feature_dir=str(ref_root),
        setting='visa_to_mvtec',
    )

    with pytest.raises(FileNotFoundError, match='Incomplete ResAD reference features'):
        validate_official_inputs(args)


def test_build_reference_feature_manifest_hashes_complete_feature_tree(tmp_path):
    root = tmp_path / 'mvtec_4shot'
    for class_name in ['bottle', 'cable']:
        class_dir = root / class_name
        class_dir.mkdir(parents=True)
        for level in range(1, 4):
            (class_dir / f'layer{level}.npy').write_bytes(f'{class_name}-{level}'.encode('utf-8'))

    manifest = build_reference_feature_manifest(
        root,
        class_names=['bottle', 'cable'],
    )

    assert manifest['root_dir'] == str(root.resolve())
    assert manifest['file_count'] == 6
    assert manifest['total_bytes'] > 0
    assert len(manifest['sha256']) == 64


def test_parse_resad_official_metrics_extracts_latest_merged_average():
    text = '\n'.join([
        'Epoch[0/1]: train_loss = 5.9987',
        'image AUROC = 0.649',
        'pixel AUROC = 0.898',
        'image AP = 0.822',
        'pixel AP = 0.253',
        'image AUROC = 0.701',
    ])

    metrics = parse_resad_official_metrics(text)

    assert metrics == {
        'train_loss': 5.9987,
        'image_auroc': 0.701,
        'pixel_auroc': 0.898,
        'image_ap': 0.822,
        'pixel_ap': 0.253,
    }


def test_parse_resad_official_metrics_supports_historical_merged_format():
    text = '\n'.join([
        'Epoch[0/10]: train_loss: 6.089682710080302',
        '(Merged) Average Image AUC | AP | F1_Score: 0.700 | 0.848 | 0.862, '
        'Average Pixel AUC | AP | F1_Score | AUPRO: 0.849 | 0.278 | 0.306 | -1.000',
    ])

    metrics = parse_resad_official_metrics(text)

    assert metrics == {
        'train_loss': 6.089682710080302,
        'image_auroc': 0.700,
        'image_ap': 0.848,
        'image_f1': 0.862,
        'pixel_auroc': 0.849,
        'pixel_ap': 0.278,
        'pixel_f1': 0.306,
        'aupro': -1.0,
    }


def test_ensure_resad_few_shot_dir_extracts_dataset_from_archive(tmp_path):
    archive_path = tmp_path / 'ResAD-data.zip'
    with zipfile.ZipFile(archive_path, 'w') as zf:
        zf.writestr('data/4shot/mvtec/bottle/train/good/000.png', b'img')
        zf.writestr('data/4shot/mvtec/bottle/train/good/001.png', b'img')
        zf.writestr('data/4shot/visa/candle/train/good/000.png', b'img')

    few_shot_dir = ensure_resad_few_shot_dir(
        tmp_path / 'pretrained' / 'resad' / 'data' / '4shot' / 'mvtec',
        archive_path=archive_path,
        dataset='mvtec',
    )

    assert few_shot_dir.exists()
    assert (few_shot_dir / 'bottle' / 'train' / 'good' / '000.png').is_file()
    assert not (few_shot_dir / 'candle').exists()


def test_patch_official_dataloader_num_workers_overrides_upstream_value():
    class DummyModule:
        @staticmethod
        def DataLoader(*args, **kwargs):
            return kwargs

    patch_official_dataloader_num_workers(DummyModule, num_workers=0)
    kwargs = DummyModule.DataLoader(dataset='dataset', batch_size=32, num_workers=8)

    assert kwargs['num_workers'] == 0
    assert kwargs['batch_size'] == 32


def test_patch_official_train_reference_tensor_cache_preserves_output_shape(tmp_path, monkeypatch):
    visa_root = tmp_path / 'visa'
    for class_name in ['candle', 'capsules']:
        class_dir = visa_root / class_name / 'train' / 'good'
        class_dir.mkdir(parents=True)
        for index in range(4):
            _write_fake_rgb(class_dir / f'{index:03d}.png', size=(32, 32))

    monkeypatch.setitem(sys.modules, 'utils', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'datasets.mvtec', SimpleNamespace(MVTEC=SimpleNamespace(CLASS_NAMES=['bottle'])))
    monkeypatch.setitem(
        sys.modules,
        'datasets.visa',
        SimpleNamespace(VISA=SimpleNamespace(CLASS_NAMES=['candle', 'capsules'])),
    )

    class DummyEncoder:
        def __call__(self, images):
            batch = images.shape[0]
            device = images.device
            return [
                torch.ones(batch, 256, 56, 56, device=device),
                torch.ones(batch, 512, 28, 28, device=device),
                torch.ones(batch, 1024, 14, 14, device=device),
            ]

    module = SimpleNamespace(
        SETTINGS={'visa_to_mvtec': {'seen': ['candle', 'capsules']}},
        get_mc_reference_features=lambda *args, **kwargs: None,
    )

    patch_official_train_reference_tensor_cache(
        module,
        setting='visa_to_mvtec',
        train_dataset_dir=visa_root,
        preload=True,
    )

    outputs = module.get_mc_reference_features(
        DummyEncoder(),
        str(visa_root),
        ['candle', 'capsules', 'candle'],
        'cpu',
        num_shot=2,
    )

    assert set(outputs.keys()) == {'candle', 'capsules'}
    assert outputs['candle'][0].shape == (2 * 56 * 56, 256)
    assert outputs['candle'][1].shape == (2 * 28 * 28, 512)
    assert outputs['candle'][2].shape == (2 * 14 * 14, 1024)


def test_initialize_official_seeds_prefers_module_init_function():
    calls = {}

    module = SimpleNamespace(init_seeds=lambda seed: calls.setdefault('seed', seed))
    initialize_official_seeds(module, seed=42)

    assert calls == {'seed': 42}


def test_initialize_official_seeds_falls_back_to_utils_module(monkeypatch):
    calls = {}
    monkeypatch.setitem(sys.modules, 'utils', SimpleNamespace(init_seeds=lambda seed: calls.setdefault('seed', seed)))

    module = SimpleNamespace()
    initialize_official_seeds(module, seed=123)

    assert calls == {'seed': 123}


def test_resad_official_eval_dry_run_imports_without_optional_memae(tmp_path):
    ref_root = tmp_path / '.refs' / 'ResAD'
    ref_root.mkdir(parents=True)
    train_root = tmp_path / 'data' / 'visa'
    train_root.mkdir(parents=True)
    test_root = tmp_path / 'data' / 'mvtec'
    test_root.mkdir(parents=True)
    feature_root = tmp_path / 'pretrained' / 'resad' / 'ref_features' / 'w50' / 'mvtec_4shot'
    for class_name in [
        'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
        'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper',
    ]:
        class_dir = feature_root / class_name
        class_dir.mkdir(parents=True)
        for filename in ('layer1.npy', 'layer2.npy', 'layer3.npy'):
            (class_dir / filename).write_bytes(b'npy')

    config_path = tmp_path / 'resad_official_test.py'
    config_path.write_text(
        '\n'.join([
            f"ref_repo_root = {ref_root.as_posix()!r}",
            "setting = 'visa_to_mvtec'",
            f"train_dataset_dir = {train_root.as_posix()!r}",
            f"test_dataset_dir = {test_root.as_posix()!r}",
            f"test_ref_feature_dir = {feature_root.as_posix()!r}",
            f"work_dir = {(tmp_path / 'runs').as_posix()!r}",
            "device = 'cpu'",
        ]),
        encoding='utf-8',
    )

    completed = subprocess.run(
        [sys.executable, 'tools/resad_official_eval.py', str(config_path), '--dry-run'],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert '"setting": "visa_to_mvtec"' in completed.stdout
    summary_path = tmp_path / 'runs' / 'official_summary.json'
    command_log_path = tmp_path / 'runs' / 'command.log'
    assert summary_path.is_file()
    assert command_log_path.is_file()

    payload = json.loads(summary_path.read_text(encoding='utf-8'))
    assert payload['status'] == 'dry_run'
    assert payload['returncode'] == 0
    assert payload['checkpoint_count'] == 0
    assert payload['no_checkpoint_expected'] is True
    assert payload['ref_feature_manifest']['file_count'] == 45


def test_resad_official_eval_backfills_summary_from_existing_log(tmp_path):
    ref_root = tmp_path / '.refs' / 'ResAD'
    ref_root.mkdir(parents=True)
    train_root = tmp_path / 'data' / 'visa'
    train_root.mkdir(parents=True)
    test_root = tmp_path / 'data' / 'mvtec'
    test_root.mkdir(parents=True)
    feature_root = tmp_path / 'pretrained' / 'resad' / 'ref_features' / 'w50' / 'mvtec_4shot'
    for class_name in [
        'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
        'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper',
    ]:
        class_dir = feature_root / class_name
        class_dir.mkdir(parents=True)
        for filename in ('layer1.npy', 'layer2.npy', 'layer3.npy'):
            (class_dir / filename).write_bytes(b'npy')

    work_dir = tmp_path / 'runs'
    work_dir.mkdir(parents=True)
    (work_dir / 'command.log').write_text(
        '\n'.join([
            '{"stage": "full"}',
            'Epoch[0/100]: train_loss: 6.091063825899765',
            '(Merged) Average Image AUC | AP | F1_Score: 0.700 | 0.848 | 0.862, Average Pixel AUC | AP | F1_Score | AUPRO: 0.849 | 0.278 | 0.306 | -1.000',
            '[autorun] returncode=0',
        ]),
        encoding='utf-8',
    )

    config_path = tmp_path / 'resad_official_test.py'
    config_path.write_text(
        '\n'.join([
            f"ref_repo_root = {ref_root.as_posix()!r}",
            "setting = 'visa_to_mvtec'",
            f"train_dataset_dir = {train_root.as_posix()!r}",
            f"test_dataset_dir = {test_root.as_posix()!r}",
            f"test_ref_feature_dir = {feature_root.as_posix()!r}",
            f"work_dir = {work_dir.as_posix()!r}",
            "device = 'cpu'",
        ]),
        encoding='utf-8',
    )

    completed = subprocess.run(
        [sys.executable, 'tools/resad_official_eval.py', str(config_path), '--backfill-summary'],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads((work_dir / 'official_summary.json').read_text(encoding='utf-8'))
    assert payload['status'] == 'completed'
    assert payload['summary_origin'] == 'backfill_from_existing_log'
    assert payload['parsed_metrics']['image_auroc'] == 0.7
    assert payload['parsed_metrics']['pixel_auroc'] == 0.849
    assert payload['checkpoint_count'] == 0
    assert payload['launch_header']['stage'] == 'full'
    assert 'config_overrides_from_log' not in payload


def test_resad_official_eval_backfill_preserves_logged_cfg_overrides(tmp_path):
    ref_root = tmp_path / '.refs' / 'ResAD'
    ref_root.mkdir(parents=True)
    train_root = tmp_path / 'data' / 'visa'
    train_root.mkdir(parents=True)
    test_root = tmp_path / 'data' / 'mvtec'
    test_root.mkdir(parents=True)
    feature_root = tmp_path / 'pretrained' / 'resad' / 'ref_features' / 'w50' / 'mvtec_4shot'
    for class_name in [
        'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
        'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper',
    ]:
        class_dir = feature_root / class_name
        class_dir.mkdir(parents=True)
        for filename in ('layer1.npy', 'layer2.npy', 'layer3.npy'):
            (class_dir / filename).write_bytes(b'npy')

    work_dir = tmp_path / 'runs'
    work_dir.mkdir(parents=True)
    (work_dir / 'command.log').write_text(
        '\n'.join([
            json.dumps({
                'stage': 'mid',
                'work_dir': str(work_dir),
                'cmd': [
                    sys.executable,
                    'tools/resad_official_eval.py',
                    '--prepare-ref-features',
                    '--work-dir',
                    str(work_dir),
                    '--cfg-options',
                    'epochs=10',
                    'eval_freq=10',
                    'batch_size=32',
                    'cache_train_ref_tensors=True',
                    'train_dataset_dir=/tmp/resad_official_data/visa',
                    'test_dataset_dir=/dev/shm/baoiad_data/mvtec_ad',
                ],
            }),
            'Epoch[0/10]: train_loss: 6.089682710080302',
            '(Merged) Average Image AUC | AP | F1_Score: 0.712 | 0.853 | 0.856, Average Pixel AUC | AP | F1_Score | AUPRO: 0.837 | 0.206 | 0.272 | -1.000',
            '[autorun] returncode=0',
        ]),
        encoding='utf-8',
    )

    config_path = tmp_path / 'resad_official_test.py'
    config_path.write_text(
        '\n'.join([
            f"ref_repo_root = {ref_root.as_posix()!r}",
            "setting = 'visa_to_mvtec'",
            f"train_dataset_dir = {train_root.as_posix()!r}",
            f"test_dataset_dir = {test_root.as_posix()!r}",
            f"test_ref_feature_dir = {feature_root.as_posix()!r}",
            f"work_dir = {work_dir.as_posix()!r}",
            "device = 'cpu'",
        ]),
        encoding='utf-8',
    )

    completed = subprocess.run(
        [sys.executable, 'tools/resad_official_eval.py', str(config_path), '--backfill-summary'],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads((work_dir / 'official_summary.json').read_text(encoding='utf-8'))
    assert payload['launch_header']['cmd'] == [
        sys.executable,
        'tools/resad_official_eval.py',
        '--prepare-ref-features',
        '--work-dir',
        str(work_dir),
        '--cfg-options',
        'epochs=10',
        'eval_freq=10',
        'batch_size=32',
        'cache_train_ref_tensors=True',
        'train_dataset_dir=/tmp/resad_official_data/visa',
        'test_dataset_dir=/dev/shm/baoiad_data/mvtec_ad',
    ]
    assert payload['config_overrides_from_log'] == {
        'epochs': 10,
        'eval_freq': 10,
        'batch_size': 32,
        'cache_train_ref_tensors': True,
        'train_dataset_dir': '/tmp/resad_official_data/visa',
        'test_dataset_dir': '/dev/shm/baoiad_data/mvtec_ad',
    }


def test_patch_freia_soft_permutation_rvs_uses_fast_sampler(monkeypatch):
    import FrEIA.modules.all_in_one_block as all_in_one_block

    original_rvs = all_in_one_block.special_ortho_group.rvs
    monkeypatch.setattr(all_in_one_block.special_ortho_group, 'rvs', lambda dim, *args, **kwargs: np.full((dim, dim), 7.0, dtype=np.float32))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)

    calls = {}

    def fake_sampler(dim, device=None):
        calls['dim'] = dim
        calls['device'] = device
        return np.eye(dim, dtype=np.float32)

    monkeypatch.setattr(freia_utils, '_sample_special_ortho_torch', fake_sampler)
    freia_utils.patch_freia_soft_permutation_rvs(min_dim=4, device='cuda:3')

    patched_rvs = all_in_one_block.special_ortho_group.rvs
    assert patched_rvs is not original_rvs
    assert np.allclose(patched_rvs(4), np.eye(4, dtype=np.float32))
    assert np.allclose(patched_rvs(2), np.full((2, 2), 7.0, dtype=np.float32))
    assert calls == {'dim': 4, 'device': 'cuda:3'}


def test_patch_timm_local_pretrained_uses_cached_checkpoint(monkeypatch, tmp_path):
    import timm

    checkpoint_dir = tmp_path / 'hub' / 'checkpoints'
    checkpoint_dir.mkdir(parents=True)
    checkpoint_path = checkpoint_dir / 'wide_resnet50_2-95faca4d.pth'
    checkpoint_path.write_bytes(b'ckpt')

    calls = {}

    class FakeModel:
        def load_state_dict(self, state_dict, strict=False):
            calls['state_dict'] = state_dict
            calls['strict'] = strict

    def fake_create_model(model_name, *args, pretrained=False, **kwargs):
        calls['model_name'] = model_name
        calls['pretrained'] = pretrained
        return FakeModel()

    monkeypatch.setattr(torch.hub, 'get_dir', lambda: str(tmp_path / 'hub'))
    monkeypatch.setattr(torch, 'load', lambda path, map_location='cpu': {'weight': torch.tensor([1.0])})
    monkeypatch.setattr(timm, 'create_model', fake_create_model)

    patch_timm_local_pretrained()
    model = timm.create_model('wide_resnet50_2', pretrained=True, features_only=True)

    assert isinstance(model, FakeModel)
    assert calls['model_name'] == 'wide_resnet50_2'
    assert calls['pretrained'] is False
    assert calls['strict'] is False
    assert 'weight' in calls['state_dict']

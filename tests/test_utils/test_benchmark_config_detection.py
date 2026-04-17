"""Tests for benchmark config metadata detection."""

import importlib.util
from functools import lru_cache
from pathlib import Path

from baoiad.utils.graphcore_alignment import graphcore_strict_alignment_violations


ROOT = Path(__file__).resolve().parents[2]


class _FakePopen:
    """Minimal subprocess.Popen stub for benchmark tests."""

    def __init__(self, cmd, *, stdout_text='', stderr_text='', returncode=0, on_start=None):
        self.cmd = cmd
        self.pid = 424242
        self.returncode = returncode
        self._stdout_text = stdout_text
        self._stderr_text = stderr_text
        if on_start is not None:
            on_start(cmd)

    def communicate(self, timeout=None):
        del timeout
        return self._stdout_text, self._stderr_text


@lru_cache(maxsize=1)
def _load_benchmark_module():
    benchmark_path = ROOT / 'tools' / 'benchmark.py'
    spec = importlib.util.spec_from_file_location('baoiad_benchmark', benchmark_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_aaclip_benchmark_uses_multi_class_flag():
    benchmark = _load_benchmark_module()
    config_path = ROOT / 'configs' / 'aaclip' / 'aaclip_vitl14_336_518_mvtec_strict.py'
    assert benchmark.is_multi_class_config(str(config_path)) is True


def test_aaclip_benchmark_prefers_strict_config():
    benchmark = _load_benchmark_module()
    config_path = benchmark.find_config('aaclip')
    assert config_path.endswith('aaclip_vitl14_336_518_mvtec_strict.py')


def test_aaclip_benchmark_keeps_train_root_and_is_eval_only():
    benchmark = _load_benchmark_module()
    config_path = ROOT / 'configs' / 'aaclip' / 'aaclip_vitl14_336_518_mvtec_strict.py'
    assert benchmark.keep_train_data_root(str(config_path)) is True
    assert benchmark.is_eval_only_config(str(config_path)) is True


def test_anomalyclip_benchmark_prefers_strict_config():
    benchmark = _load_benchmark_module()
    config_path = benchmark.find_config('anomalyclip')
    assert config_path.endswith('anomalyclip_vitl14_336_518_mvtec_strict.py')


def test_patchcore_benchmark_prefers_strict_config():
    benchmark = _load_benchmark_module()
    config_path = benchmark.find_config('patchcore')
    assert config_path.endswith('patchcore_wrn50_256_mvtec_strict.py')


def test_patchcore_benchmark_remains_single_class():
    benchmark = _load_benchmark_module()
    config_path = ROOT / 'configs' / 'patchcore' / 'patchcore_wrn50_256_mvtec_strict.py'
    assert benchmark.is_multi_class_config(str(config_path)) is False


def test_patchcore_benchmark_uses_direct_runner():
    benchmark = _load_benchmark_module()
    config_path = ROOT / 'configs' / 'patchcore' / 'patchcore_wrn50_256_mvtec_strict.py'
    assert benchmark._should_use_direct_runner(str(config_path)) is True


def test_padim_benchmark_remains_single_class():
    benchmark = _load_benchmark_module()
    config_path = ROOT / 'configs' / 'padim' / 'padim_wrn50_256_mvtec.py'
    assert benchmark.is_multi_class_config(str(config_path)) is False


def test_padim_config_freezes_official_wrn50_hparams():
    benchmark = _load_benchmark_module()
    config_path = ROOT / 'configs' / 'padim' / 'padim_wrn50_256_mvtec.py'
    cfg = benchmark._load_config(str(config_path))

    assert cfg.model['type'] == 'PaDiMDetector'
    assert cfg.model['backbone']['type'] == 'TIMMBackbone'
    assert cfg.model['backbone']['model_name'] == 'wide_resnet50_2'
    assert tuple(cfg.model['backbone']['out_indices']) == (1, 2, 3)
    assert cfg.model['backbone']['frozen'] is True
    assert cfg.model['sigma'] == 4.0
    assert cfg.train_dataloader.batch_size == 32
    assert cfg.train_cfg.max_epochs == 1


def test_resad_benchmark_uses_multi_class_flag():
    benchmark = _load_benchmark_module()
    config_path = ROOT / 'configs' / 'resad' / 'resad_wrn50_256_mvtec.py'
    assert benchmark.is_multi_class_config(str(config_path)) is True




def test_resad_strict_config_freezes_official_hparams():
    benchmark = _load_benchmark_module()
    config_path = ROOT / 'configs' / 'resad' / 'resad_wrn50_256_mvtec_strict.py'
    cfg = benchmark._load_config(str(config_path))

    assert cfg.benchmark_multi_class is True
    assert cfg.randomness == {'seed': 42, 'deterministic': False}
    assert cfg.train_dataloader.batch_size == 32
    assert cfg.train_dataloader.dataset['split'] == 'train'
    assert cfg.train_dataloader.dataset['multi_class'] is True
    assert cfg.test_dataloader.batch_size == 1
    assert cfg.test_dataloader.dataset['split'] == 'test'
    assert cfg.test_dataloader.dataset['multi_class'] is True
    assert cfg.model['type'] == 'ResADDetector'
    assert cfg.model['backbone']['model_name'] == 'wide_resnet50_2'
    assert tuple(cfg.model['backbone']['out_indices']) == (1, 2, 3)
    assert cfg.model['input_size'] == 224
    assert cfg.model['num_embeddings'] == 1536
    assert cfg.model['coupling_layers'] == 10
    assert cfg.model['clamp_alpha'] == 1.9
    assert cfg.model['strict_ref_features'] is True
    assert cfg.train_cfg['type'] == 'ResADOfficialTrainLoop'
    assert cfg.train_cfg['max_epochs'] == 100
    assert cfg.train_cfg['val_interval'] == 10
    assert cfg.train_cfg['first_stage_epochs'] == 10
    assert cfg.train_cfg['N_batch'] == 8192


def test_saaplus_benchmark_prefers_strict_config():
    benchmark = _load_benchmark_module()
    config_path = benchmark.find_config('saaplus')
    assert config_path.endswith('saaplus_400_mvtec_strict.py')


def test_dinomaly_benchmark_prefers_strict_config():
    benchmark = _load_benchmark_module()
    config_path = benchmark.find_config('dinomaly')
    assert config_path.endswith('dinomaly_392_mvtec_strict.py')


def test_dinomaly_strict_benchmark_is_multi_class():
    benchmark = _load_benchmark_module()
    config_path = ROOT / 'configs' / 'dinomaly' / 'dinomaly_392_mvtec_strict.py'
    assert benchmark.is_multi_class_config(str(config_path)) is True


def test_dinomaly_strict_benchmark_preserves_workers():
    benchmark = _load_benchmark_module()
    config_path = ROOT / 'configs' / 'dinomaly' / 'dinomaly_392_mvtec_strict.py'
    assert benchmark.keep_dataloader_workers(str(config_path)) is True
    assert benchmark.keep_checkpoint_hooks(str(config_path)) is True
    assert benchmark.resume_existing_benchmark(str(config_path)) is True


def test_dinomaly_strict_config_freezes_official_muad_hparams():
    benchmark = _load_benchmark_module()
    config_path = ROOT / 'configs' / 'dinomaly' / 'dinomaly_392_mvtec_strict.py'
    cfg = benchmark._load_config(str(config_path))

    assert cfg.benchmark_multi_class is True
    assert cfg.benchmark_keep_dataloader_workers is True
    assert cfg.benchmark_preserve_checkpoint_hooks is True
    assert cfg.benchmark_resume_existing is True
    assert cfg.train_dataloader.batch_size == 16
    assert cfg.train_dataloader.dataset['multi_class'] is True
    assert len(cfg.train_dataloader.dataset['cls_names']) == 15
    assert cfg.train_pipeline[0]['type'] == 'LoadImage'
    assert cfg.train_pipeline[2]['type'] == 'ResizeAD'
    assert cfg.train_pipeline[2]['size'] == 448
    assert cfg.train_pipeline[2]['official_pil'] is True
    assert cfg.train_pipeline[3] == {'type': 'CenterCrop', 'size': 392}
    assert cfg.model['backbone']['type'] == 'DinomalyEncoder'
    assert cfg.model['backbone']['encoder_name'] == 'dinov2reg_vit_base_14'
    assert cfg.model['predict_map_size'] == 256
    assert cfg.model['image_score_max_ratio'] == 0.01
    assert cfg.optim_wrapper['optimizer']['type'] == 'StableAdamW'
    assert cfg.optim_wrapper['optimizer']['lr'] == 2e-3
    assert cfg.optim_wrapper['optimizer']['eps'] == 1e-10
    assert cfg.optim_wrapper['clip_grad']['max_norm'] == 0.1
    assert cfg.param_scheduler[0]['type'] == 'WarmCosineLR'
    assert cfg.param_scheduler[0]['warmup_iters'] == 100
    assert cfg.param_scheduler[0]['final_value'] == 2e-4
    assert cfg.train_cfg['by_epoch'] is False
    assert cfg.train_cfg['max_iters'] == 10000
    assert cfg.train_cfg['val_begin'] == 5000
    assert cfg.train_cfg['val_interval'] == 5000
    assert cfg.test_evaluator['resize_mask'] == 256


def test_dfm_benchmark_prefers_strict_config():
    benchmark = _load_benchmark_module()
    config_path = benchmark.find_config('dfm')
    assert config_path.endswith('dfm_256_mvtec_strict.py')


def test_efficientad_benchmark_prefers_strict_config():
    benchmark = _load_benchmark_module()
    config_path = benchmark.find_config('efficientad')
    assert config_path.endswith('efficientad_256_mvtec_strict.py')


def test_efficientad_strict_config_matches_anomalib_padding():
    benchmark = _load_benchmark_module()
    cfg = benchmark._load_config(str(ROOT / 'configs' / 'efficientad' / 'efficientad_256_mvtec_strict.py'))
    assert cfg.model['padding'] is False
    assert cfg.model['pad_maps'] is True


def test_efficientad_strict_benchmark_preserves_workers_and_resume():
    benchmark = _load_benchmark_module()
    config_path = ROOT / 'configs' / 'efficientad' / 'efficientad_256_mvtec_strict.py'
    assert benchmark.keep_dataloader_workers(str(config_path)) is True
    assert benchmark.keep_checkpoint_hooks(str(config_path)) is True
    assert benchmark.resume_existing_benchmark(str(config_path)) is True


def test_graphcore_benchmark_prefers_official_vig_config():
    benchmark = _load_benchmark_module()
    config_path = benchmark.find_config('graphcore')
    assert config_path.endswith('graphcore_vig_ti_224_mvtec_strict.py')


def test_memseg_benchmark_prefers_strict_config():
    benchmark = _load_benchmark_module()
    config_path = benchmark.find_config('memseg')
    assert config_path.endswith('memseg_rn18_256_mvtec_strict.py')


def test_musc_benchmark_prefers_strict_config():
    benchmark = _load_benchmark_module()
    config_path = benchmark.find_config('musc')
    assert config_path.endswith('musc_vitl14_336_518_mvtec_strict.py')


def test_musc_strict_config_freezes_official_zero_shot_hparams():
    benchmark = _load_benchmark_module()
    cfg = benchmark._load_config(str(ROOT / 'configs' / 'musc' / 'musc_vitl14_336_518_mvtec_strict.py'))

    assert cfg.benchmark_multi_class is False
    assert cfg.benchmark_eval_only is True
    assert cfg.benchmark_timeout == 14400
    assert cfg.randomness == {'seed': 42, 'deterministic': False}
    assert cfg.train_dataloader.batch_size == 4
    assert cfg.test_dataloader.batch_size == 4
    assert cfg.train_dataloader.dataset['multi_class'] is True
    assert cfg.test_dataloader.dataset['multi_class'] is True
    assert cfg.model['type'] == 'MuScDetector'
    assert cfg.model['backbone']['type'] == 'MuScCLIPBackbone'
    assert cfg.model['backbone']['model_name'] == 'ViT-L-14-336'
    assert cfg.model['backbone']['pretrained'] == 'openai'
    assert cfg.model['backbone']['require_ref_open_clip'] is True
    assert cfg.model['feature_layers'] == [5, 11, 17, 23]
    assert cfg.model['r_list'] == [1, 3, 5]
    assert cfg.model['topmin_min'] == 0.0
    assert cfg.model['topmin_max'] == 0.3
    assert cfg.model['k_list'] == [1, 2, 3]
    assert cfg.train_pipeline[2]['type'] == 'OpenCLIPPreprocessAD'
    assert cfg.train_pipeline[2]['size'] == 518
    assert cfg.train_cfg['max_epochs'] == 1


def test_memae_benchmark_prefers_official_video_strict_config():
    benchmark = _load_benchmark_module()
    config_path = benchmark.find_config('memae')
    assert config_path.endswith('memae_ucsdped2_256_official.py')


def test_memae_strict_benchmark_path_is_explicitly_closed_for_generic_runner():
    benchmark = _load_benchmark_module()
    strict_config = benchmark.find_config('memae')
    assert strict_config.endswith('memae_ucsdped2_256_official.py')
    reason = benchmark.closed_strict_benchmark_reason('memae', strict_config)
    assert reason is not None

    old_mvtec_strict = ROOT / 'configs' / 'memae' / 'memae_wrn50_256_mvtec.py'
    assert benchmark.closed_strict_benchmark_reason('memae', str(old_mvtec_strict)) is not None

    adapted_config = ROOT / 'configs' / 'memae' / 'memae_wrn50_256_mvtec_adapted.py'
    assert benchmark.closed_strict_benchmark_reason('memae', str(adapted_config)) is None


def test_memseg_strict_config_freezes_official_hparams():
    benchmark = _load_benchmark_module()
    config_path = ROOT / 'configs' / 'memseg' / 'memseg_rn18_256_mvtec_strict.py'
    cfg = benchmark._load_config(str(config_path))

    assert cfg.benchmark_multi_class is False
    assert cfg.train_dataloader.batch_size == 8
    assert cfg.train_dataloader.num_workers == 0
    assert cfg.train_dataloader.sampler['type'] == 'PersistentShuffleSampler'
    assert cfg.train_dataloader.sampler['shuffle'] is True
    assert cfg.train_dataloader.sampler['seed'] == 42
    assert cfg.train_dataloader.sampler['round_up'] is False
    assert cfg.test_dataloader.batch_size == 8
    assert cfg.train_cfg['by_epoch'] is False
    assert cfg.train_cfg['max_iters'] == 5000
    assert cfg.train_cfg['val_interval'] == 100
    assert cfg.model['nb_memory_sample'] == 30
    assert cfg.model['alternate_anomaly_sampling'] is True
    assert cfg.model['require_texture_source'] is True
    assert cfg.model['use_imgaug'] is True
    assert cfg.model['memory_bank_seed'] == 42
    assert cfg.model['anomaly_source_resize'] == 288
    assert cfg.model['anomaly_source_crop'] == 256
    assert cfg.model['backbone']['frozen'] is False
    assert tuple(cfg.model['backbone']['frozen_names']) == ('layer1', 'layer2', 'layer3')
    assert cfg.model['backbone']['frozen_names_eval'] is False
    assert [hook['type'] for hook in cfg.custom_hooks] == ['MemSegStrictTrainHook', 'MemoryBankHook']
    assert cfg.benchmark_result_selector == {
        'mode': 'best_balanced',
        'metrics': ['image_auroc', 'pixel_auroc', 'aupro'],
        'tie_break_metric': 'image_ap',
    }
    assert cfg.randomness == {'seed': 42, 'deterministic': False}


def test_graphcore_official_config_freezes_mainline_hparams():
    benchmark = _load_benchmark_module()
    config_path = ROOT / 'configs' / 'graphcore' / 'graphcore_vig_ti_224_mvtec_strict.py'
    cfg = benchmark._load_config(str(config_path))

    assert cfg.benchmark_multi_class is False
    assert cfg.train_dataloader.batch_size == 32
    assert cfg.test_dataloader.batch_size == 1
    assert cfg.train_cfg.max_epochs == 1
    assert cfg.train_dataloader.sampler['type'] == 'OpenIADSubsetRandomSampler'
    assert cfg.train_dataloader.sampler['seed'] == 66
    assert cfg.model['backbone']['type'] == 'GraphCoreViGBackbone'
    assert cfg.model['backbone']['model_name'] == 'vig_ti_224_gelu'
    assert cfg.model['n_neighbours'] == 9
    assert cfg.model['sampler_percentage'] == 0.001
    assert cfg.model['layer_num_1'] == 3
    assert cfg.model['layer_num_2'] == 4
    assert tuple(cfg.model['input_size']) == (224, 224)
    assert cfg.model['image_score_mode'] == 'raw_max'
    assert cfg.model['image_score_mode_overrides'] == {}
    assert cfg.model['random_seed'] == 66
    assert cfg.model['coreset_initial_index'] == 0
    assert cfg.randomness['seed'] == 66
    assert graphcore_strict_alignment_violations(cfg.model) == []
    assert benchmark.strict_alignment_guard_errors(str(config_path)) == []


def test_graphcore_benchmark_uses_explicit_order_file_when_available():
    benchmark = _load_benchmark_module()
    order_dir = ROOT / 'runs' / 'alignment' / 'graphcore_orders'
    order_dir.mkdir(parents=True, exist_ok=True)
    order_file = order_dir / 'transistor.json'
    original = order_file.read_text(encoding='utf-8') if order_file.exists() else None
    order_file.write_text('{"indices": [0, 1, 2]}', encoding='utf-8')

    try:
        options = benchmark._graphcore_train_order_cfg_options(
            str(ROOT / 'configs' / 'graphcore' / 'graphcore_vig_ti_224_mvtec_strict.py'),
            'transistor',
        )
        assert "train_dataloader.sampler.type='ExplicitOrderSampler'" in options
        assert "train_dataloader.sampler.index_file='runs/alignment/graphcore_orders/transistor.json'" in options
    finally:
        if original is None:
            order_file.unlink(missing_ok=True)
        else:
            order_file.write_text(original, encoding='utf-8')


def test_uniad_benchmark_prefers_strict_config():
    benchmark = _load_benchmark_module()
    config_path = benchmark.find_config('uniad')
    assert config_path.endswith('uniad_wrn50_256_mvtec_strict.py')


def test_uniad_strict_config_freezes_effnet_b4_muad_mainline():
    benchmark = _load_benchmark_module()
    config_path = ROOT / 'configs' / 'uniad' / 'uniad_wrn50_256_mvtec_strict.py'
    cfg = benchmark._load_config(str(config_path))

    assert cfg.benchmark_multi_class is True
    assert cfg.benchmark_keep_dataloader_workers is True
    assert cfg.benchmark_preserve_checkpoint_hooks is True
    assert cfg.train_dataloader.dataset['multi_class'] is True
    assert cfg.test_dataloader.dataset['multi_class'] is True
    assert cfg.model['type'] == 'UniADDetector'
    assert cfg.model['backbone']['type'] == 'TIMMBackbone'
    assert cfg.model['backbone']['model_name'] == 'tf_efficientnet_b4'
    assert tuple(cfg.model['backbone']['out_indices']) == (0, 1, 2, 3)
    assert cfg.model['backbone']['frozen'] is True
    assert cfg.model['image_score_mode'] == 'pooled_topk_mean'
    assert cfg.model['image_score_topk'] == 128


def test_uninet_benchmark_prefers_strict_config():
    benchmark = _load_benchmark_module()
    config_path = benchmark.find_config('uninet')
    assert config_path.endswith('uninet_256_mvtec_strict.py')


def test_uninet_strict_benchmark_is_single_class():
    benchmark = _load_benchmark_module()
    config_path = ROOT / 'configs' / 'uninet' / 'uninet_256_mvtec_strict.py'
    assert benchmark.is_multi_class_config(str(config_path)) is False


def test_uninet_strict_benchmark_preserves_workers_and_best_selector():
    benchmark = _load_benchmark_module()
    config_path = ROOT / 'configs' / 'uninet' / 'uninet_256_mvtec_strict.py'
    assert benchmark.keep_dataloader_workers(str(config_path)) is True
    assert benchmark.keep_checkpoint_hooks(str(config_path)) is True
    assert benchmark.benchmark_result_selector(str(config_path)) == {
        'mode': 'best',
        'metric': 'image_auroc',
    }


def test_regad_strict_config_requires_official_support_set():
    benchmark = _load_benchmark_module()
    config_path = ROOT / 'configs' / 'regad' / 'regad_wrn50_256_mvtec_strict.py'
    cfg = benchmark._load_config(str(config_path))

    assert cfg.support_set_root.endswith('data/regad_official/support_set')
    assert cfg.strict_require_official_support_set is True
    assert benchmark.keep_dataloader_workers(str(config_path)) is True
    assert benchmark.resume_existing_benchmark(str(config_path)) is True
    assert benchmark.benchmark_train_script(str(config_path)).endswith('tools/train_regad_strict.py')
    assert benchmark.benchmark_result_selector(str(config_path)) == {
        'mode': 'best_balanced',
        'metrics': ['image_auroc', 'pixel_auroc'],
    }


def test_regad_benchmark_prefers_strict_config():
    benchmark = _load_benchmark_module()
    config_path = benchmark.find_config('regad')
    assert config_path.endswith('regad_wrn50_256_mvtec_strict.py')


def test_uflow_benchmark_prefers_strict_config():
    benchmark = _load_benchmark_module()
    config_path = benchmark.find_config('uflow')
    assert config_path.endswith('uflow_mcait_448_mvtec_strict.py')


def test_uflow_strict_config_freezes_official_mainline_hparams():
    benchmark = _load_benchmark_module()
    config_path = ROOT / 'configs' / 'uflow' / 'uflow_mcait_448_mvtec_strict.py'
    cfg = benchmark._load_config(str(config_path))

    assert benchmark.keep_dataloader_workers(str(config_path)) is True
    assert benchmark.keep_checkpoint_hooks(str(config_path)) is True
    assert benchmark.benchmark_result_selector(str(config_path)) == {
        'mode': 'best',
        'metric': 'pixel_auroc',
    }
    assert cfg.train_dataloader.batch_size == 8
    assert cfg.test_dataloader.batch_size == 5
    assert cfg.train_cfg.max_epochs == 200
    assert cfg.model['backbone'] == 'mcait'
    assert tuple(cfg.model['input_size']) == (448, 448)
    assert cfg.default_hooks['checkpoint']['save_best'] == 'ad/pixel_auroc'
    assert any(hook['type'] == 'UFlowStrictTrainHook' for hook in cfg.custom_hooks)


def test_uflow_strict_category_cfg_options_follow_official_yaml():
    benchmark = _load_benchmark_module()
    config_path = ROOT / 'configs' / 'uflow' / 'uflow_mcait_448_mvtec_strict.py'
    bottle_options = benchmark.benchmark_category_cfg_options(str(config_path), 'bottle')
    tile_options = benchmark.benchmark_category_cfg_options(str(config_path), 'tile')

    assert 'train_dataloader.batch_size=23' in bottle_options
    assert 'optim_wrapper.optimizer.lr=0.00011289990475381853' in bottle_options
    assert 'train_dataloader.batch_size=30' in tile_options
    assert 'optim_wrapper.optimizer.lr=0.006045754779719109' in tile_options


def test_uflow_strict_benchmark_is_not_iter_based():
    benchmark = _load_benchmark_module()
    config_path = ROOT / 'configs' / 'uflow' / 'uflow_mcait_448_mvtec_strict.py'
    assert benchmark.is_iter_based(str(config_path)) is False


def test_anomalydino_benchmark_prefers_strict_config():
    benchmark = _load_benchmark_module()
    config_path = benchmark.find_config('anomalydino')
    assert config_path.endswith('anomalydino_vitb14_448_mvtec_strict.py')


def test_anomalydino_strict_config_freezes_official_preprocess():
    benchmark = _load_benchmark_module()
    config_path = ROOT / 'configs' / 'anomalydino' / 'anomalydino_vitb14_448_mvtec_strict.py'
    cfg = benchmark._load_config(str(config_path))

    assert cfg.model['preprocess'] == 'agnostic'
    assert cfg.model['mask_ref_images'] is False
    assert cfg.model['few_shot'] == 1
    assert cfg.model['few_shot_seed'] == 0
    assert cfg.randomness['seed'] == 0
    resize_step = cfg.train_dataloader.dataset['pipeline'][2]
    assert resize_step['type'] == 'ResizeAD'
    assert resize_step['size'] == 448
    assert resize_step['keep_ratio'] is True
    assert cfg.train_dataloader.batch_size == 1


def test_invad_benchmark_prefers_strict_config():
    benchmark = _load_benchmark_module()
    config_path = benchmark.find_config('invad')
    assert config_path.endswith('invad_wrn50_256_mvtec_strict.py')


def test_invad_strict_benchmark_is_multi_class():
    benchmark = _load_benchmark_module()
    config_path = ROOT / 'configs' / 'invad' / 'invad_wrn50_256_mvtec_strict.py'
    assert benchmark.is_multi_class_config(str(config_path)) is True


def test_invad_strict_config_freezes_official_mainline_hparams():
    benchmark = _load_benchmark_module()
    config_path = ROOT / 'configs' / 'invad' / 'invad_wrn50_256_mvtec_strict.py'
    cfg = benchmark._load_config(str(config_path))

    assert cfg.benchmark_result_selector == {
        'mode': 'best_per_metric',
        'metrics': ['image_auroc', 'pixel_auroc'],
    }
    assert cfg.benchmark_keep_dataloader_workers is True
    assert cfg.benchmark_preserve_checkpoint_hooks is True
    assert cfg.benchmark_resume_existing is True
    assert cfg.benchmark_timeout == 108000
    assert cfg.train_cfg['max_epochs'] == 300
    assert cfg.train_cfg['val_interval'] == 30
    assert cfg.train_dataloader.batch_size == 32
    assert cfg.test_dataloader.batch_size == 32
    assert cfg.train_dataloader.dataset['shuffle_train_data'] is True
    assert cfg.param_scheduler[0]['step_size'] == 240
    assert cfg.param_scheduler[0]['gamma'] == 0.1
    assert cfg.default_hooks['checkpoint']['interval'] == 10
    assert cfg.default_hooks['checkpoint']['max_keep_ckpts'] == 3
    assert cfg.default_hooks['checkpoint']['save_last'] is True

    backbone = cfg.model['backbone']
    assert backbone['type'] == 'TIMMBackbone'
    assert backbone['model_name'] == 'wide_resnet50_2'
    assert backbone['checkpoint_path'].endswith('pretrained/wide_resnet50_racm-8234f177.pth')
    assert tuple(backbone['out_indices']) == (1, 2, 3)
    assert cfg.model['out_cha'] == 64
    assert cfg.model['latent_channel_size'] == 16
    assert cfg.model['gaussian_sigma'] == 4.0


def test_invad_strict_benchmark_preserves_workers():
    benchmark = _load_benchmark_module()
    config_path = ROOT / 'configs' / 'invad' / 'invad_wrn50_256_mvtec_strict.py'
    assert benchmark.keep_dataloader_workers(str(config_path)) is True
    assert benchmark.keep_checkpoint_hooks(str(config_path)) is True
    assert benchmark.resume_existing_benchmark(str(config_path)) is True
    assert benchmark.benchmark_timeout(str(config_path), 7200) == 108000


def test_parse_metrics_can_select_best_per_metric_snapshot():
    benchmark = _load_benchmark_module()
    output = '\n'.join([
        'Epoch(val) [30][54/54] ad/image_auroc: 0.8000 ad/pixel_auroc: 0.9300',
        'Epoch(val) [60][54/54] ad/image_auroc: 0.8800 ad/pixel_auroc: 0.9100',
        'Epoch(val) [90][54/54] ad/image_auroc: 0.8600 ad/pixel_auroc: 0.9500',
    ])
    metrics = benchmark.parse_metrics(
        output,
        selector=dict(mode='best_per_metric', metrics=['image_auroc', 'pixel_auroc']),
    )
    assert metrics['image_auroc'] == 0.88
    assert metrics['pixel_auroc'] == 0.95


def test_vitad_benchmark_prefers_muad_config():
    benchmark = _load_benchmark_module()
    config_path = benchmark.find_config('vitad')
    assert config_path.endswith('vitad_256_mvtec_muad.py')


def test_vitad_muad_benchmark_is_multi_class():
    benchmark = _load_benchmark_module()
    config_path = ROOT / 'configs' / 'vitad' / 'vitad_256_mvtec_muad.py'
    assert benchmark.is_multi_class_config(str(config_path)) is True


def test_vitad_muad_config_freezes_reference_model_hparams():
    benchmark = _load_benchmark_module()
    config_path = ROOT / 'configs' / 'vitad' / 'vitad_256_mvtec_muad.py'
    cfg = benchmark._load_config(str(config_path))
    assert cfg.model['encoder_name'] == 'vit_small_patch16_224.dino' or cfg.model['encoder_name'] == 'vit_small_patch16_224_dino'
    assert tuple(cfg.model['teachers']) == (3, 6, 9)
    assert tuple(cfg.model['neck']) == (12,)
    assert tuple(cfg.model['students']) == (3, 6, 9)
    assert cfg.model['decoder_depth'] == 9
    assert cfg.model['fusion_mul'] == 1
    assert cfg.model['gaussian_sigma'] == 4.0
    assert cfg.train_dataloader.dataset['shuffle_train_data'] is True
    assert cfg.optim_wrapper['constructor'] == 'baoiad.ViTADOptimWrapperConstructor'
    assert cfg.benchmark_train_script == 'tools/train_vitad_exact_order.py'
    assert cfg.benchmark_keep_dataloader_workers is True


def test_vitad_strict_config_freezes_protocol_guards():
    benchmark = _load_benchmark_module()
    config_path = ROOT / 'configs' / 'vitad' / 'vitad_256_mvtec_strict.py'
    cfg = benchmark._load_config(str(config_path))

    assert cfg.benchmark_multi_class is True
    assert cfg.benchmark_keep_dataloader_workers is True
    assert cfg.benchmark_preserve_checkpoint_hooks is True
    assert cfg.benchmark_result_selector['mode'] == 'last'
    assert cfg.benchmark_train_script == 'tools/train_vitad_exact_order.py'
    assert cfg.env_cfg['cudnn_benchmark'] is True
    assert cfg.train_dataloader.sampler['type'] == 'PersistentShuffleSampler'
    assert cfg.train_dataloader.sampler['seed'] == 42
    assert cfg.train_dataloader['drop_last'] is True
    assert cfg.train_dataloader['pin_memory'] is True
    assert cfg.train_dataloader['persistent_workers'] is False
    assert cfg.test_dataloader['pin_memory'] is True
    assert cfg.test_dataloader['persistent_workers'] is False
    assert cfg.train_cfg['val_begin'] == 10
    assert cfg.train_cfg['val_interval'] == 10
    assert cfg.param_scheduler == []
    assert cfg.optim_wrapper['constructor'] == 'baoiad.ViTADOptimWrapperConstructor'
    assert any(hook['type'] == 'ViTADStrictTrainHook' for hook in cfg.custom_hooks)


def test_vitad_benchmark_uses_exact_order_train_script_and_preserves_workers(monkeypatch):
    benchmark = _load_benchmark_module()
    captured = {}

    def fake_popen(cmd, stdout, stderr, text, cwd, env, start_new_session):
        del stdout, stderr, text, cwd, env, start_new_session
        captured['cmd'] = cmd
        return _FakePopen(cmd)

    monkeypatch.setattr(benchmark.subprocess, 'Popen', fake_popen)

    config_path = ROOT / 'configs' / 'vitad' / 'vitad_256_mvtec_muad.py'
    benchmark.run_method(
        str(config_path),
        data_root='data/mvtec_ad',
        category='bottle',
        device='cuda',
        epochs=10,
        batch_size=None,
        work_dir='runs/test',
        timeout=60,
        multi_class=True,
    )

    assert captured['cmd'][1].endswith('tools/train_vitad_exact_order.py')
    cfg_options = captured['cmd'][captured['cmd'].index('--cfg-options') + 1:]
    assert 'train_dataloader.num_workers=0' not in cfg_options
    assert 'test_dataloader.num_workers=0' not in cfg_options
    assert 'val_dataloader.num_workers=0' not in cfg_options


def test_vitad_single_class_benchmark_remains_single_class():
    benchmark = _load_benchmark_module()
    config_path = ROOT / 'configs' / 'vitad' / 'vitad_wrn50_256_mvtec.py'
    assert benchmark.is_multi_class_config(str(config_path)) is False


def test_spade_benchmark_prefers_strict_config():
    benchmark = _load_benchmark_module()
    config_path = benchmark.find_config('spade')
    assert config_path.endswith('spade_wrn50_224_mvtec_strict.py')


def test_simplenet_benchmark_prefers_strict_config():
    benchmark = _load_benchmark_module()
    config_path = benchmark.find_config('simplenet')
    assert config_path.endswith('simplenet_wrn50_288_mvtec_strict.py')


def test_simplenet_strict_config_freezes_official_hparams():
    benchmark = _load_benchmark_module()
    config_path = ROOT / 'configs' / 'simplenet' / 'simplenet_wrn50_288_mvtec_strict.py'
    cfg = benchmark._load_config(str(config_path))

    assert cfg.benchmark_result_selector == {'mode': 'best', 'metric': 'image_auroc'}
    assert cfg.benchmark_keep_dataloader_workers is True
    assert cfg.randomness['seed'] == 0
    assert cfg.train_dataloader.batch_size == 8
    assert cfg.test_dataloader.batch_size == 8
    assert cfg.train_cfg.max_epochs == 160
    assert cfg.train_cfg.val_interval == 4
    assert cfg.param_scheduler == []
    assert cfg.test_evaluator['normalize_image_scores'] is True
    assert cfg.test_evaluator['normalize_pred_maps'] == 'batch_broadcast'

    resize_step = cfg.train_dataloader.dataset['pipeline'][2]
    crop_step = cfg.train_dataloader.dataset['pipeline'][3]
    assert resize_step['type'] == 'ResizeAD'
    assert resize_step['size'] == 329
    assert crop_step['type'] == 'CenterCrop'
    assert crop_step['size'] == 288

    assert cfg.model['strict'] is True
    assert cfg.model['image_size'] == 288
    assert cfg.model['gaussian_sigma'] == 4.0
    assert cfg.model['noise_std'] == 0.015
    assert cfg.model['dsc_margin'] == 0.5
    assert cfg.model['pre_proj'] == 1
    assert cfg.model['backbone']['backbone_name'] == 'wide_resnet50_2'
    assert tuple(cfg.model['backbone']['out_indices']) == (2, 3)

    assert cfg.optim_wrapper['constructor'] == 'SimpleNetOptimWrapperConstructor'
    assert cfg.optim_wrapper['projection']['optimizer']['type'] == 'AdamW'
    assert cfg.optim_wrapper['discriminator']['optimizer']['type'] == 'Adam'


def test_destseg_benchmark_prefers_rn18_strict_config():
    benchmark = _load_benchmark_module()
    config_path = benchmark.find_config('destseg')
    assert config_path.endswith('destseg_rn18_256_mvtec_strict.py')


def test_destseg_strict_config_uses_official_teacher_init():
    benchmark = _load_benchmark_module()
    config_path = ROOT / 'configs' / 'destseg' / 'destseg_rn18_256_mvtec_strict.py'
    cfg = benchmark._load_config(str(config_path))

    assert cfg.model['teacher_pretrained'] is True
    assert 'teacher_checkpoint_path' not in cfg.model
    assert cfg.train_cfg['max_iters'] == 5000
    assert cfg.model['de_st_steps'] == 1000


def test_destseg_strict_config_uses_soft_mask_resize_then_threshold():
    benchmark = _load_benchmark_module()
    config_path = ROOT / 'configs' / 'destseg' / 'destseg_rn18_256_mvtec_strict.py'
    cfg = benchmark._load_config(str(config_path))

    train_pipeline = cfg.train_dataloader.dataset['pipeline']
    test_pipeline = cfg.test_dataloader.dataset['pipeline']

    assert train_pipeline[0]['type'] == 'LoadImage'
    assert train_pipeline[0]['backend'] == 'pil'

    assert test_pipeline[0] == dict(type='LoadImage', backend='pil')
    assert test_pipeline[1] == dict(type='LoadMask', backend='pil', to_binary=False)
    assert test_pipeline[2]['type'] == 'ResizeAD'
    assert test_pipeline[2]['backend'] == 'pillow'
    assert test_pipeline[2]['mask_interpolation'] == 'bilinear'
    assert test_pipeline[3] == dict(type='ThresholdMask', threshold=0.5)


def test_cutpaste_benchmark_prefers_strict_config():
    benchmark = _load_benchmark_module()
    config_path = benchmark.find_config('cutpaste')
    assert config_path.endswith('cutpaste_rn18_256_mvtec_strict.py')


def test_stfpm_benchmark_prefers_strict_config():
    benchmark = _load_benchmark_module()
    config_path = benchmark.find_config('stfpm')
    assert config_path.endswith('stfpm_rn18_256_mvtec_strict.py')


def test_stfpm_strict_config_freezes_official_protocol():
    benchmark = _load_benchmark_module()
    config_path = ROOT / 'configs' / 'stfpm' / 'stfpm_rn18_256_mvtec_strict.py'
    cfg = benchmark._load_config(str(config_path))

    assert benchmark.is_multi_class_config(str(config_path)) is False
    assert benchmark.keep_checkpoint_hooks(str(config_path)) is True
    assert benchmark.resume_existing_benchmark(str(config_path)) is True
    assert benchmark.benchmark_test_after_train(str(config_path)) is True
    assert benchmark.benchmark_checkpoint_source(str(config_path)) == 'best'
    assert benchmark.benchmark_timeout(str(config_path), 7200) == 14400
    assert cfg.randomness['seed'] == 0
    assert cfg.train_dataloader.batch_size == 32
    assert cfg.val_dataloader.batch_size == 32
    assert cfg.test_dataloader.batch_size == 1
    assert cfg.train_dataloader.dataset['train_val_split_ratio'] == 0.2
    assert cfg.train_dataloader.dataset['train_val_split_subset'] == 'train'
    assert cfg.val_dataloader.dataset['train_val_split_subset'] == 'val'
    assert cfg.val_evaluator['type'] == 'AnomalyMapMeanMetric'
    assert cfg.default_hooks['checkpoint']['save_best'] == 'ad/score_mean'
    assert cfg.default_hooks['checkpoint']['rule'] == 'less'
    assert cfg.default_hooks['checkpoint']['interval'] == 1000000
    assert cfg.model['reference_impl'] == 'official'
    assert cfg.optim_wrapper['optimizer']['weight_decay'] == 1e-4


def test_glass_benchmark_timeout_uses_config_minimum():
    benchmark = _load_benchmark_module()
    config_path = ROOT / 'configs' / 'glass' / 'glass_wrn50_288_mvtec_strict.py'

    assert benchmark.benchmark_timeout(str(config_path), 3600) == 7200
    assert benchmark.benchmark_timeout(str(config_path), 14400) == 14400


def test_csflow_strict_config_freezes_runtime_and_benchmark_resume_policy():
    benchmark = _load_benchmark_module()
    config_path = ROOT / 'configs' / 'csflow' / 'csflow_256_mvtec_strict.py'
    cfg = benchmark._load_config(str(config_path))

    assert benchmark.keep_checkpoint_hooks(str(config_path)) is True
    assert benchmark.keep_dataloader_workers(str(config_path)) is True
    assert benchmark.resume_existing_benchmark(str(config_path)) is True
    assert benchmark.benchmark_test_after_train(str(config_path)) is True
    assert benchmark.benchmark_checkpoint_source(str(config_path)) == 'best'
    assert benchmark.benchmark_timeout(str(config_path), 7200) == 14400
    assert cfg.benchmark_result_selector == dict(mode='best', metric='image_auroc')
    assert cfg.default_hooks['checkpoint']['save_best'] == 'ad/image_auroc'
    assert cfg.default_hooks['checkpoint']['rule'] == 'greater'
    assert cfg.default_hooks['checkpoint']['save_last'] is True
    assert cfg.train_dataloader.num_workers == 8
    assert cfg.val_dataloader.num_workers == 8
    assert cfg.test_dataloader.num_workers == 8
    assert cfg.train_dataloader.persistent_workers is False
    assert cfg.val_dataloader.persistent_workers is False
    assert cfg.test_dataloader.persistent_workers is False
    assert cfg.train_cfg['max_epochs'] == 240
    assert cfg.train_cfg['val_interval'] == 60
    # Original paper uses constant lr (no param_scheduler)
    assert not hasattr(cfg, 'param_scheduler')


def test_csflow_benchmark_uses_strict_mainline():
    benchmark = _load_benchmark_module()
    config_path = benchmark.find_config('csflow')

    assert config_path.endswith('csflow_256_mvtec_strict.py')


def test_rd_benchmark_prefers_strict_config():
    benchmark = _load_benchmark_module()
    config_path = benchmark.find_config('rd')
    assert config_path.endswith('rd_wrn50_256_mvtec_strict.py')


def test_rd_strict_benchmark_remains_single_class():
    benchmark = _load_benchmark_module()
    config_path = ROOT / 'configs' / 'rd' / 'rd_wrn50_256_mvtec_strict.py'
    assert benchmark.is_multi_class_config(str(config_path)) is False


def test_rd_strict_benchmark_preserves_workers():
    benchmark = _load_benchmark_module()
    config_path = ROOT / 'configs' / 'rd' / 'rd_wrn50_256_mvtec_strict.py'
    assert benchmark.keep_dataloader_workers(str(config_path)) is True


def test_cutpaste_strict_config_freezes_official_resnet18_hparams():
    benchmark = _load_benchmark_module()
    config_path = ROOT / 'configs' / 'cutpaste' / 'cutpaste_rn18_256_mvtec_strict.py'
    cfg = benchmark._load_config(str(config_path))

    assert cfg.train_dataloader.batch_size == 64
    assert cfg.train_cfg.max_iters == 256
    assert cfg.train_cfg.val_interval == 10
    assert cfg.model['freeze_iters'] == 20
    assert tuple(cfg.model['head_dims']) == (512, 128)
    assert cfg.model['backbone']['type'] == 'RawBackbone'
    assert cfg.model['backbone']['backbone_name'] == 'resnet18'
    assert cfg.model['force_backbone_eval_while_frozen'] is False
    assert cfg.model['num_classes'] == 3


def test_pyramidflow_benchmark_prefers_strict_config():
    benchmark = _load_benchmark_module()
    config_path = benchmark.find_config('pyramidflow')
    assert config_path.endswith('pyramidflow_resnet18_1024_mvtec_strict.py')


def test_pyramidflow_strict_config_freezes_published_resnet18_protocol():
    benchmark = _load_benchmark_module()
    config_path = ROOT / 'configs' / 'pyramidflow' / 'pyramidflow_resnet18_1024_mvtec_strict.py'
    cfg = benchmark._load_config(str(config_path))

    assert cfg.benchmark_multi_class is False
    assert cfg.benchmark_keep_dataloader_workers is True
    assert list(cfg.benchmark_summary_categories) == [
        'bottle', 'cable', 'capsule', 'carpet', 'hazelnut', 'leather',
        'pill', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper',
    ]
    assert cfg.train_dataloader.batch_size == 2
    assert cfg.train_dataloader.dataset['shuffle_train_data'] is True
    resize_step = cfg.train_dataloader.dataset['pipeline'][2]
    assert resize_step['type'] == 'ResizeAD'
    assert resize_step['size'] == 1024
    normalize_step = cfg.train_dataloader.dataset['pipeline'][3]
    assert normalize_step['type'] == 'NormalizeAD'
    assert cfg.test_evaluator['resize_mask'] == 256
    assert cfg.test_evaluator['image_score_field'] == 'pred_score_max'
    assert cfg.optim_wrapper['optimizer']['lr'] == 2e-4
    assert cfg.optim_wrapper['optimizer']['eps'] == 1e-4
    assert tuple(cfg.optim_wrapper['optimizer']['betas']) == (0.5, 0.9)
    assert cfg.optim_wrapper['clip_grad']['max_norm'] == 1.0
    assert cfg.param_scheduler == []
    assert cfg.model['pyramid_downsample_mode'] == 'maxpool'
    assert cfg.model['predict_resize_to_input'] is False
    assert list(cfg.model['template_pipeline']) == list(cfg.test_pipeline)


def test_pyramidflow_fnf_strict_config_uses_spatial_max_image_score():
    benchmark = _load_benchmark_module()
    config_path = ROOT / 'configs' / 'pyramidflow' / 'pyramidflow_fnf_256_mvtec_strict.py'
    cfg = benchmark._load_config(str(config_path))

    assert cfg.test_evaluator['image_score_field'] == 'pred_score_max'
    assert cfg.model['pyramid_downsample_mode'] == 'maxpool'
    assert list(cfg.model['template_pipeline']) == list(cfg.test_pipeline)


def test_pyramidflow_strict_config_exposes_official_summary_subset():
    benchmark = _load_benchmark_module()
    config_path = ROOT / 'configs' / 'pyramidflow' / 'pyramidflow_resnet18_1024_mvtec_strict.py'
    assert benchmark.configured_benchmark_categories(str(config_path)) is None
    assert benchmark.configured_benchmark_summary_categories(str(config_path)) == [
        'bottle', 'cable', 'capsule', 'carpet', 'hazelnut', 'leather',
        'pill', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper',
    ]


def test_pyramidflow_strict_category_cfg_options_are_empty_after_texture_proxy_promotion():
    benchmark = _load_benchmark_module()
    config_path = ROOT / 'configs' / 'pyramidflow' / 'pyramidflow_resnet18_1024_mvtec_strict.py'
    carpet_options = benchmark.benchmark_category_cfg_options(str(config_path), 'carpet')
    bottle_options = benchmark.benchmark_category_cfg_options(str(config_path), 'bottle')

    assert carpet_options == []
    assert bottle_options == []


def test_ast_benchmark_prefers_strict_config():
    benchmark = _load_benchmark_module()
    config_path = benchmark.find_config('ast')
    assert config_path.endswith('ast_effnet_b5_768_mvtec_strict.py')


def test_ast_strict_config_uses_two_stage_train_script():
    benchmark = _load_benchmark_module()
    config_path = ROOT / 'configs' / 'ast' / 'ast_effnet_b5_768_mvtec_strict.py'
    train_script = benchmark.benchmark_train_script(str(config_path))
    assert train_script.endswith('tools/train_ast.py')


def test_supersimplenet_benchmark_prefers_strict_config():
    benchmark = _load_benchmark_module()
    config_path = benchmark.find_config('supersimplenet')
    assert config_path.endswith('supersimplenet_256_mvtec_strict.py')


def test_supersimplenet_strict_benchmark_rescales_multistep_scheduler():
    benchmark = _load_benchmark_module()
    config_path = ROOT / 'configs' / 'supersimplenet' / 'supersimplenet_256_mvtec_strict.py'
    cfg = benchmark._load_config(str(config_path))

    assert cfg.benchmark_rescale_epoch_schedulers is True
    assert cfg.param_scheduler[0]['type'] == 'MultiStepLR'
    assert cfg.param_scheduler[0]['milestones'] == [240, 270]


def test_supersimplenet_epoch_override_rescales_scheduler(monkeypatch):
    benchmark = _load_benchmark_module()
    captured = {}

    def fake_popen(cmd, stdout, stderr, text, cwd, env, start_new_session):
        del stdout, stderr, text, cwd, env, start_new_session
        captured['cmd'] = cmd
        return _FakePopen(cmd)

    monkeypatch.setattr(benchmark.subprocess, 'Popen', fake_popen)

    config_path = ROOT / 'configs' / 'supersimplenet' / 'supersimplenet_256_mvtec_strict.py'
    benchmark.run_method(
        str(config_path),
        data_root='data/mvtec_ad',
        category='bottle',
        device='cuda',
        epochs=20,
        batch_size=None,
        work_dir='runs/test',
        timeout=60,
        multi_class=False,
    )

    cfg_options = captured['cmd'][captured['cmd'].index('--cfg-options') + 1:]
    assert 'train_cfg.max_epochs=20' in cfg_options
    assert 'param_scheduler.0.milestones=[16, 18]' in cfg_options


def test_anomalyclip_strict_benchmark_preserves_train_data_root():
    benchmark = _load_benchmark_module()
    config_path = ROOT / 'configs' / 'anomalyclip' / 'anomalyclip_vitl14_336_518_mvtec_strict.py'
    assert benchmark.keep_train_data_root(str(config_path)) is True


def test_anomalyclip_strict_benchmark_is_trainable():
    benchmark = _load_benchmark_module()
    config_path = ROOT / 'configs' / 'anomalyclip' / 'anomalyclip_vitl14_336_518_mvtec_strict.py'
    assert benchmark.is_eval_only_config(str(config_path)) is False
    assert benchmark.benchmark_test_after_train(str(config_path)) is True


def test_anomalyclip_strict_config_freezes_official_trainable_hparams():
    benchmark = _load_benchmark_module()
    cfg = benchmark._load_config(
        str(ROOT / 'configs' / 'anomalyclip' / 'anomalyclip_vitl14_336_518_mvtec_strict.py')
    )

    assert cfg.benchmark_multi_class is True
    assert cfg.benchmark_keep_train_data_root is True
    assert cfg.benchmark_test_after_train is True
    assert cfg.benchmark_checkpoint_source == 'last'
    assert cfg.randomness == {'seed': 111, 'deterministic': True}
    assert cfg.train_dataloader.batch_size == 8
    assert cfg.train_dataloader.num_workers == 0
    assert cfg.train_dataloader.persistent_workers is False
    assert cfg.train_dataloader.dataset['type'] == 'VisADataset'
    assert cfg.train_dataloader.dataset['split'] == 'test'
    assert cfg.train_dataloader.dataset['multi_class'] is True
    assert tuple(cfg.train_dataloader.dataset['pipeline'][2]['size']) == (518, 518)
    assert cfg.test_dataloader.batch_size == 1
    assert cfg.test_dataloader.num_workers == 0
    assert cfg.test_dataloader.persistent_workers is False
    assert cfg.test_dataloader.dataset['type'] == 'MVTecADDataset'
    assert cfg.test_dataloader.dataset['split'] == 'test'
    assert cfg.test_dataloader.dataset['multi_class'] is True
    assert tuple(cfg.test_dataloader.dataset['pipeline'][2]['size']) == (518, 518)
    assert cfg.model['type'] == 'AnomalyCLIPOfficialDetector'
    assert cfg.model['clip_model'] == 'ViT-L/14@336px'
    assert cfg.model['image_size'] == 518
    assert cfg.model['features_list'] == [24]
    assert cfg.model['feature_map_layer'] == [0, 1, 2, 3]
    assert cfg.model['prompt_depth'] == 9
    assert cfg.model['prompt_length'] == 12
    assert cfg.model['prompt_text_length'] == 4
    assert cfg.model['temperature'] == 0.07
    assert cfg.model['gaussian_sigma'] == 4.0
    assert cfg.model['dpam_layer'] == 20
    assert cfg.model['official_checkpoint'] is None
    assert cfg.optim_wrapper['optimizer']['type'] == 'Adam'
    assert cfg.optim_wrapper['optimizer']['lr'] == 1e-3
    assert cfg.optim_wrapper['optimizer']['betas'] == (0.5, 0.999)
    assert cfg.param_scheduler == []
    assert cfg.default_hooks['checkpoint']['interval'] == 1
    assert cfg.default_hooks['checkpoint']['save_last'] is True
    assert cfg.train_cfg['max_epochs'] == 15
    assert cfg.train_cfg['val_begin'] == 16
    assert cfg.train_cfg['val_interval'] == 16


def test_differnet_benchmark_uses_best_image_auroc_selector():
    benchmark = _load_benchmark_module()
    config_path = ROOT / 'configs' / 'differnet' / 'differnet_alexnet_256_mvtec.py'
    assert benchmark.benchmark_result_selector(str(config_path)) == {
        'mode': 'best',
        'metric': 'image_auroc',
    }


def test_parse_metrics_can_select_best_snapshot():
    benchmark = _load_benchmark_module()
    output = '\n'.join([
        'Epoch(val) [24][5/5] ad/image_auroc: 0.7000 ad/pixel_auroc: 0.6000',
        'Epoch(val) [48][5/5] ad/image_auroc: 0.8200 ad/pixel_auroc: 0.5500',
        'Epoch(val) [72][5/5] ad/image_auroc: 0.7100 ad/pixel_auroc: 0.6500',
    ])
    metrics = benchmark.parse_metrics(
        output,
        selector=dict(mode='best', metric='image_auroc'),
    )
    assert metrics['image_auroc'] == 0.82
    assert metrics['pixel_auroc'] == 0.55


def test_benchmark_epoch_override_also_forces_final_validation(monkeypatch):
    benchmark = _load_benchmark_module()
    captured = {}

    def fake_popen(cmd, stdout, stderr, text, cwd, env, start_new_session):
        del stdout, stderr, text, cwd, start_new_session
        captured['cmd'] = cmd
        captured['env'] = env
        return _FakePopen(cmd)

    monkeypatch.setattr(benchmark.subprocess, 'Popen', fake_popen)

    config_path = ROOT / 'configs' / 'differnet' / 'differnet_alexnet_256_mvtec.py'
    benchmark.run_method(
        str(config_path),
        data_root='data/mvtec_ad',
        category='bottle',
        device='cuda',
        epochs=1,
        batch_size=None,
        work_dir='runs/test',
        timeout=60,
        multi_class=False,
    )

    cfg_options = captured['cmd'][captured['cmd'].index('--cfg-options') + 1:]
    assert 'train_cfg.max_epochs=1' in cfg_options
    assert 'train_cfg.val_interval=1' in cfg_options
    assert 'train_cfg.val_begin=1' in cfg_options


def test_benchmark_applies_uflow_category_cfg_options(monkeypatch):
    benchmark = _load_benchmark_module()
    captured = {}

    def fake_popen(cmd, stdout, stderr, text, cwd, env, start_new_session):
        del stdout, stderr, text, cwd, start_new_session
        captured['cmd'] = cmd
        captured['env'] = env
        return _FakePopen(cmd)

    monkeypatch.setattr(benchmark.subprocess, 'Popen', fake_popen)

    config_path = ROOT / 'configs' / 'uflow' / 'uflow_mcait_448_mvtec_strict.py'
    benchmark.run_method(
        str(config_path),
        data_root='data/mvtec_ad',
        category='bottle',
        device='cuda',
        epochs=None,
        batch_size=None,
        work_dir='runs/test',
        timeout=60,
        multi_class=False,
    )

    cfg_options = captured['cmd'][captured['cmd'].index('--cfg-options') + 1:]
    assert 'train_dataloader.batch_size=23' in cfg_options
    assert 'optim_wrapper.optimizer.lr=0.00011289990475381853' in cfg_options


def test_stfpm_benchmark_trains_then_tests_best_checkpoint(monkeypatch, tmp_path):
    benchmark = _load_benchmark_module()
    captured = {'cmds': []}

    def fake_popen(cmd, stdout, stderr, text, cwd, env, start_new_session):
        del stdout, stderr, text, cwd, env, start_new_session
        captured['cmds'].append(cmd)
        work_dir = Path(cmd[cmd.index('--work-dir') + 1])
        work_dir.mkdir(parents=True, exist_ok=True)
        if cmd[1].endswith('tools/train.py'):
            (work_dir / 'best_ad_score_mean_epoch_1.pth').write_text('fake checkpoint')
            return _FakePopen(
                cmd,
                stdout_text='Epoch(val) [1][1/1] ad/score_mean: 0.1234\n',
            )
        return _FakePopen(
            cmd,
            stdout_text='Epoch(test) [1][1/1] ad/image_auroc: 0.9876 ad/pixel_auroc: 0.8765\n',
        )

    monkeypatch.setattr(benchmark.subprocess, 'Popen', fake_popen)

    config_path = ROOT / 'configs' / 'stfpm' / 'stfpm_rn18_256_mvtec_strict.py'
    metrics, _ = benchmark.run_method(
        str(config_path),
        data_root='data/mvtec_ad',
        category='bottle',
        device='cuda',
        epochs=1,
        batch_size=None,
        work_dir=str(tmp_path / 'stfpm'),
        timeout=60,
        multi_class=False,
    )

    assert metrics['image_auroc'] == 0.9876
    assert metrics['pixel_auroc'] == 0.8765
    assert len(captured['cmds']) == 2
    assert captured['cmds'][0][1].endswith('tools/train.py')
    assert captured['cmds'][1][1].endswith('tools/test.py')
    assert captured['cmds'][1][3].endswith('best_ad_score_mean_epoch_1.pth')


def test_dinomaly_preserve_workers_skips_worker_clamp(monkeypatch):
    benchmark = _load_benchmark_module()
    captured = {}

    def fake_popen(cmd, stdout, stderr, text, cwd, env, start_new_session):
        del stdout, stderr, text, cwd, env, start_new_session
        captured['cmd'] = cmd
        return _FakePopen(cmd)

    monkeypatch.setattr(benchmark.subprocess, 'Popen', fake_popen)

    config_path = ROOT / 'configs' / 'dinomaly' / 'dinomaly_392_mvtec_strict.py'
    benchmark.run_method(
        str(config_path),
        data_root='data/mvtec_ad',
        category='bottle',
        device='cuda',
        epochs=None,
        batch_size=None,
        work_dir='runs/test',
        timeout=60,
        multi_class=False,
    )

    cfg_options = captured['cmd'][captured['cmd'].index('--cfg-options') + 1:]
    assert 'train_dataloader.num_workers=0' not in cfg_options
    assert 'test_dataloader.num_workers=0' not in cfg_options
    assert 'val_dataloader.num_workers=0' not in cfg_options


def test_destseg_benchmark_disables_compile(monkeypatch):
    benchmark = _load_benchmark_module()
    captured = {}

    def fake_popen(cmd, stdout, stderr, text, cwd, env, start_new_session):
        del stdout, stderr, text, cwd, start_new_session
        captured['cmd'] = cmd
        captured['env'] = env
        return _FakePopen(cmd)

    monkeypatch.setattr(benchmark.subprocess, 'Popen', fake_popen)

    config_path = ROOT / 'configs' / 'destseg' / 'destseg_rn18_256_mvtec_strict.py'
    benchmark.run_method(
        str(config_path),
        data_root='data/mvtec_ad',
        category='bottle',
        device='cuda',
        epochs=None,
        batch_size=None,
        work_dir='runs/test',
        timeout=60,
        multi_class=False,
    )

    cfg_options = captured['cmd'][captured['cmd'].index('--cfg-options') + 1:]
    assert 'runtime_disable_compile=True' in cfg_options
    assert captured['env']['TORCH_COMPILE_DISABLE'] == '1'
    assert captured['env']['TORCHDYNAMO_DISABLE'] == '1'


def test_destseg_benchmark_preserves_workers():
    benchmark = _load_benchmark_module()
    config_path = ROOT / 'configs' / 'destseg' / 'destseg_rn18_256_mvtec_strict.py'
    assert benchmark.keep_dataloader_workers(str(config_path)) is True


def test_univad_benchmark_prefers_strict_config():
    benchmark = _load_benchmark_module()
    config_path = benchmark.find_config('univad')
    assert config_path.endswith('univad_mvtec_strict.py')


def test_prepare_subprocess_env_prefixes_repo_root(monkeypatch):
    benchmark = _load_benchmark_module()
    monkeypatch.setenv('PYTHONPATH', '/tmp/existing')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2')

    env = benchmark._prepare_subprocess_env()

    assert env['PYTHONPATH'].split(':', 1)[0] == str(ROOT)
    assert env['CUDA_VISIBLE_DEVICES'] == '2'
    assert env['OPENBLAS_NUM_THREADS'] == '1'


def test_prepare_subprocess_env_can_disable_compile():
    benchmark = _load_benchmark_module()
    env = benchmark._prepare_subprocess_env(disable_compile=True)
    assert env['TORCH_COMPILE_DISABLE'] == '1'
    assert env['TORCHDYNAMO_DISABLE'] == '1'


def test_pni_benchmark_prefers_formal_strict_config():
    benchmark = _load_benchmark_module()
    config_path = benchmark.find_config('pni')
    assert config_path.endswith('pni_wrn101_480_mvtec_strict.py')


def test_pni_strict_config_uses_keep_ratio_resize():
    benchmark = _load_benchmark_module()
    cfg = benchmark._load_config(
        str(ROOT / 'configs' / 'pni' / 'pni_wrn101_480_mvtec_strict.py')
    )
    resize_step = cfg.train_pipeline[2]
    assert resize_step['type'] == 'ResizeAD'
    assert resize_step['size'] == 512
    assert resize_step['keep_ratio'] is True
    assert cfg.train_dataloader.dataset['multi_class'] is False
    assert 'cls_names' not in cfg.train_dataloader.dataset


def test_pni_strict_config_uses_official_coreset_path():
    benchmark = _load_benchmark_module()
    cfg = benchmark._load_config(
        str(ROOT / 'configs' / 'pni' / 'pni_wrn101_480_mvtec_strict.py')
    )
    assert cfg.model['head']['approximate_coreset'] is False
    assert cfg.model['head']['distribution_size'] == 2048
    assert cfg.model['head']['mlp_val_ratio'] == 0.1
    assert cfg.randomness['seed'] == 23
    assert 'cls_names' not in cfg.test_dataloader.dataset


def test_pni_bounded_config_keeps_runtime_safe_approximate_coreset():
    benchmark = _load_benchmark_module()
    cfg = benchmark._load_config(str(ROOT / 'configs' / 'pni' / 'pni_wrn50_256_mvtec.py'))
    assert cfg.model['head']['approximate_coreset'] is True

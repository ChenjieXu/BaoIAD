"""AnomalyCLIP strict official-alignment config.

Reference freeze:
- Official repository: `zqhang/AnomalyCLIP`
- Commit: `3911738c0867544f545a076ad78f3f11d9ecbfdf`
- Runtime authority: `train.py`, `test.py`, `loss.py`, `utils.py`
- Official protocol: train prompt learner on VisA auxiliary data, then
  evaluate on MVTec AD

Important protocol notes:
- Official `train.py` uses the auxiliary dataset's `test` split because the
  reference `Dataset(...)` default is `mode='test'`.
- CLIP stays frozen in eval mode for the entire run; only the prompt learner
  is optimized.
- Official scripts save a checkpoint every epoch and report MVTec metrics after
  loading the epoch checkpoint. This config preserves that behavior.
"""

_base_ = [
    '../_base_/default_runtime.py',
]

data_root = 'data/mvtec_ad'
aux_data_root = 'data/visa'
img_size = 518

benchmark_multi_class = True
benchmark_keep_train_data_root = True
benchmark_preserve_checkpoint_hooks = True
benchmark_test_after_train = True
benchmark_checkpoint_source = 'last'
benchmark_timeout = 14400

randomness = dict(seed=111, deterministic=True)
custom_hooks = []

train_pipeline = [
    dict(type='LoadImage', backend='pil'),
    dict(type='LoadMask', backend='pil'),
    dict(
        type='ResizeAD',
        size=(img_size, img_size),
        backend='pillow',
        official_pil=True,
        mask_interpolation='nearest',
    ),
    dict(type='NormalizeAD'),
    dict(type='PackADInputs'),
]

test_pipeline = [
    dict(type='LoadImage', backend='pil'),
    dict(type='LoadMask', backend='pil'),
    dict(
        type='ResizeAD',
        size=(img_size, img_size),
        backend='pillow',
        official_pil=True,
        mask_interpolation='nearest',
    ),
    dict(type='NormalizeAD'),
    dict(type='PackADInputs'),
]

model = dict(
    type='AnomalyCLIPOfficialDetector',
    clip_model='ViT-L/14@336px',
    image_size=img_size,
    features_list=[24],
    feature_map_layer=[0, 1, 2, 3],
    prompt_depth=9,
    prompt_length=12,
    prompt_text_length=4,
    temperature=0.07,
    gaussian_sigma=4.0,
    dpam_layer=20,
    reference_root='.refs/AnomalyCLIP',
    official_checkpoint=None,
    require_official_assets=True,
    freeze_prompt_learner=False,
    enable_train_loss=True,
)

train_dataloader = dict(
    batch_size=8,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='VisADataset',
        data_root=aux_data_root,
        split='test',
        multi_class=True,
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MVTecADDataset',
        data_root=data_root,
        split='test',
        multi_class=True,
        pipeline=test_pipeline,
    ),
)

test_dataloader = val_dataloader

val_evaluator = [
    dict(
        type='AnomalyDetectionMetric',
        metrics=['image_auroc', 'pixel_auroc', 'aupro'],
    ),
]
test_evaluator = val_evaluator

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=1e-3, betas=(0.5, 0.999), weight_decay=0.0),
)

param_scheduler = []

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=15,
        save_last=True,
    ),
)

train_cfg = dict(by_epoch=True, max_epochs=15, val_begin=16, val_interval=16)
val_cfg = dict()
test_cfg = dict(type='ADTestLoop')

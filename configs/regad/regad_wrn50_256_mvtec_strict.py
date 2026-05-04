import os

_base_ = [
    '../_base_/default_runtime.py',
]

img_size = 224
shot = 4
inferences = 10
official_seed = 668
support_set_root = os.environ.get(
    'REGAD_SUPPORT_SET_ROOT',
    'data/regad_official/support_set',
)
strict_require_official_support_set = False

benchmark_train_script = 'tools/train_regad_strict.py'
benchmark_result_selector = dict(
    mode='best_balanced',
    metrics=['image_auroc', 'pixel_auroc', 'aupro'],
)
benchmark_keep_dataloader_workers = True
benchmark_preserve_checkpoint_hooks = True
benchmark_resume_existing = True

randomness = dict(seed=official_seed, deterministic=False)

train_pipeline = [
    dict(type='PackADInputs'),
]

test_pipeline = [
    dict(type='PackADInputs'),
]

train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RegADTrainDataset',
        data_root='data/mvtec_ad',
        target_cls='bottle',
        split='train',
        img_size=img_size,
        shot=shot,
        pipeline=train_pipeline,
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='RegADTestDataset',
        data_root='data/mvtec_ad',
        target_cls='bottle',
        split='test',
        img_size=img_size,
        pipeline=test_pipeline,
    ),
)

val_dataloader = test_dataloader
test_evaluator = dict(type='AnomalyDetectionMetric')
val_evaluator = test_evaluator

model = dict(
    type='RegADDetector',
    backbone='resnet18',
    sigma=4.0,
    stn_mode='rotation_scale',
    encoder_channels=256,
    data_preprocessor=dict(type='mmengine.model.ImgDataPreprocessor'),
    layers=(1, 2, 3),
    few_shot=shot,
    img_size=img_size,
    pretrained_backbone=True,
    freeze_backbone=False,
    data_root='data/mvtec_ad',
    target_cls='bottle',
)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=5e-4),
)

param_scheduler = [
    dict(type='CosineAnnealingLR', T_max=50, by_epoch=True, eta_min=0.0),
]

train_cfg = dict(by_epoch=True, max_epochs=50, val_interval=1, val_begin=1)

custom_hooks = [dict(type='MemoryBankHook')]

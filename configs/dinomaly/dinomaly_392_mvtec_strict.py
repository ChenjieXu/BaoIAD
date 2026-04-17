"""Dinomaly strict config aligned to the official MUAD MVTec entry.

Reference freeze:
- repo: guojiajeremy/Dinomaly
- commit: c5c76d01a2bd7212f1c4b7dfdad14902d0f48cfe
- runtime authority: dinomaly_mvtec_uni.py

Official protocol:
- Dinomaly-B (`dinov2reg_vit_base_14`)
- Resize(448) -> CenterCrop(392)
- MVTec AD unified multi-class training on all 15 categories
- batch_size=16, total_iters=10000
- StableAdamW(lr=2e-3, betas=(0.9, 0.999), wd=1e-4, amsgrad=True, eps=1e-10)
- warmup + cosine: warmup_iters=100, final_lr=2e-4
- grad clip max_norm=0.1
- eval image/pixel path uses resize_mask=256 + Gaussian(5, sigma=4) + top-1% mean
"""

_base_ = ['../_base_/default_runtime.py']

data_root = 'data/mvtec_ad'
resize_size = 448
img_size = 392
eval_mask_size = 256

all_categories = [
    'carpet', 'grid', 'leather', 'tile', 'wood',
    'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut',
    'pill', 'screw', 'toothbrush', 'transistor', 'zipper',
]

benchmark_multi_class = True
benchmark_keep_dataloader_workers = True
benchmark_preserve_checkpoint_hooks = True
benchmark_resume_existing = True
benchmark_result_selector = dict(mode='last')
benchmark_timeout = 43200

train_pipeline = [
    dict(type='LoadImage'),
    dict(type='LoadMask'),
    dict(type='ResizeAD', size=resize_size, backend='pillow', official_pil=True),
    dict(type='CenterCrop', size=img_size),
    dict(type='NormalizeAD'),
    dict(type='PackADInputs'),
]

test_pipeline = [
    dict(type='LoadImage'),
    dict(type='LoadMask'),
    dict(type='ResizeAD', size=resize_size, backend='pillow', official_pil=True),
    dict(type='CenterCrop', size=img_size),
    dict(type='NormalizeAD'),
    dict(type='PackADInputs'),
]

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='MVTecADDataset',
        data_root=data_root,
        split='train',
        cls_names=all_categories,
        multi_class=True,
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MVTecADDataset',
        data_root=data_root,
        split='test',
        cls_names=all_categories,
        multi_class=True,
        pipeline=test_pipeline,
    ),
)

test_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MVTecADDataset',
        data_root=data_root,
        split='test',
        cls_names=all_categories,
        multi_class=True,
        pipeline=test_pipeline,
    ),
)

val_evaluator = dict(
    type='AnomalyDetectionMetric',
    resize_mask=eval_mask_size,
    resize_gt_mask_mode='nearest',
)
test_evaluator = dict(
    type='AnomalyDetectionMetric',
    resize_mask=eval_mask_size,
    resize_gt_mask_mode='nearest',
)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=5000, max_keep_ckpts=2, save_last=True),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='ADVisualizationHook', enable=False),
)

custom_hooks = []

model = dict(
    type='DinomalyDetector',
    encoder_name='dinov2reg_vit_base_14',
    backbone=dict(
        type='DinomalyEncoder',
        encoder_name='dinov2reg_vit_base_14',
        frozen=True,
    ),
    bottleneck_dropout=0.2,
    decoder_depth=8,
    target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
    fuse_layer_encoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
    fuse_layer_decoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
    remove_class_token=False,
    loss_p_final=0.9,
    loss_schedule_steps=1000,
    loss_factor=0.1,
    predict_map_size=eval_mask_size,
    gaussian_kernel_size=5,
    gaussian_sigma=4.0,
    image_score_max_ratio=0.01,
)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='StableAdamW',
        lr=2e-3,
        betas=(0.9, 0.999),
        eps=1e-10,
        weight_decay=1e-4,
        amsgrad=True,
        clip_threshold=1.0,
    ),
    clip_grad=dict(max_norm=0.1),
)

param_scheduler = [
    dict(
        type='WarmCosineLR',
        total_iters=10000,
        warmup_iters=100,
        start_warmup_value=0.0,
        final_value=2e-4,
        begin=0,
        end=10000,
        by_epoch=False,
    ),
]

train_cfg = dict(by_epoch=False, max_iters=10000, val_begin=5000, val_interval=5000)
val_cfg = dict(type='ADValLoop')
test_cfg = dict(type='ADTestLoop')

randomness = dict(seed=1, deterministic=True)

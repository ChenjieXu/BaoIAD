# Strict GLASS config aligned to the official MVTec training script.
_base_ = ['../_base_/default_runtime.py']

benchmark_multi_class = False
benchmark_keep_dataloader_workers = True
benchmark_preserve_checkpoint_hooks = True
benchmark_result_selector = dict(
    mode='best_balanced',
    metrics=['image_auroc', 'pixel_auroc'],
    tie_break_metric='image_ap',
)
benchmark_timeout = 7200

data_root = 'data/mvtec_ad'
glass_assets_root = 'data/glass_assets/mvtec'
fg_mask_root = f'{glass_assets_root}/fg_mask'
distribution_meta_path = f'{glass_assets_root}/mvtec_distribution.xlsx'
dtd_path = 'data/dtd'
img_size = 288

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='GLASSDataset',
        data_root=data_root,
        split='train',
        cls_names=['bottle'],
        multi_class=False,
        dataset_name='mvtec',
        dtd_path=dtd_path,
        img_size=img_size,
        resize=288,
        distribution=0,
        mean=0.5,
        std=0.1,
        fg=1,
        rand_aug=1,
        downsampling=8,
        fg_mask_root=fg_mask_root,
        distribution_meta_path=distribution_meta_path,
        strict_assets_required=True,
        pipeline=[dict(type='PackGLASSInputs')],
    ),
)

test_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='GLASSDataset',
        data_root=data_root,
        split='test',
        cls_names=['bottle'],
        multi_class=False,
        dataset_name='mvtec',
        img_size=img_size,
        resize=288,
        fg=0,
        pipeline=[dict(type='PackADInputs')],
    ),
)

val_dataloader = test_dataloader
test_evaluator = dict(type='AnomalyDetectionMetric')
val_evaluator = test_evaluator

model = dict(
    type='GLASSDetector',
    strict=True,
    dtd_path=None,
    distribution=0,
    noise=0.015,
    radius=0.75,
    p=0.5,
    mining=1,
    step=20,
    limit=392,
    image_size=img_size,
    distribution_meta_path=distribution_meta_path,
)

optim_wrapper = dict(
    constructor='baoiad.GLASSOptimWrapperConstructor',
    projection=dict(
        optimizer=dict(type='Adam', lr=1e-4, weight_decay=1e-5),
    ),
    discriminator=dict(
        optimizer=dict(type='AdamW', lr=2e-4, weight_decay=1e-5),
    ),
)

param_scheduler = []

train_cfg = dict(type='GLASSTrainLoop', max_epochs=640, val_interval=1)
val_cfg = dict(type='ADValLoop')
test_cfg = dict(type='ADTestLoop')

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3, save_last=True),
)

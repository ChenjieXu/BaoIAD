"""Strict RD++ config aligned with the original official implementation."""

_base_ = ['../_base_/default_runtime.py', '../_base_/backbones/wide_resnet50_raw.py']

benchmark_timeout = 14400
benchmark_preserve_checkpoint_hooks = True
benchmark_result_selector = dict(
    mode='best_balanced',
    metrics=['image_auroc', 'pixel_auroc', 'aupro'],
    tie_break_metric='image_ap',
)

data_root = 'data/mvtec_ad'
img_size = 256

rdpp_category_epochs = dict(
    bottle=200,
    cable=240,
    capsule=300,
    carpet=10,
    grid=260,
    hazelnut=160,
    leather=10,
    metal_nut=160,
    pill=200,
    screw=280,
    tile=260,
    toothbrush=280,
    transistor=300,
    wood=100,
    zipper=300,
)

train_pipeline = [
    dict(type='LoadImage', to_float32=True),
    dict(type='ResizeAD', size=img_size, backend='cv2'),
    dict(type='ScaleNormalizeAD'),
    dict(type='GenerateRDPPNoise'),
    dict(
        type='NormalizeAD',
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        keys=('img', 'img_noise'),
    ),
    dict(type='PackRDPPInputs'),
]

test_pipeline = [
    dict(type='LoadImage', to_float32=True),
    dict(type='LoadMask'),
    dict(type='ResizeAD', size=img_size, backend='cv2'),
    dict(type='ScaleNormalizeAD'),
    dict(type='NormalizeAD', mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    dict(type='PackADInputs'),
]

train_dataloader = dict(
    batch_size=16,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='MVTecADDataset',
        data_root=data_root,
        split='train',
        cls_names=['bottle'],
        multi_class=False,
        pipeline=train_pipeline,
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MVTecADDataset',
        data_root=data_root,
        split='test',
        cls_names=['bottle'],
        multi_class=False,
        pipeline=test_pipeline,
    ),
)

val_dataloader = test_dataloader
test_evaluator = dict(type='AnomalyDetectionMetric')
val_evaluator = test_evaluator

model = dict(
    type='RDPPDetector',
    strict=True,
    backbone={{_base_.backbone}},
    smooth_sigma=4.0,
    data_preprocessor=dict(type='mmengine.model.ImgDataPreprocessor'),
)

optim_wrapper = dict(
    constructor='baoiad.RDPPOptimWrapperConstructor',
    projection=dict(
        optimizer=dict(type='Adam', lr=1e-3, betas=(0.5, 0.999)),
        accumulative_counts=2,
    ),
    distillation=dict(
        optimizer=dict(type='Adam', lr=5e-3, betas=(0.5, 0.999)),
        accumulative_counts=2,
    ),
)

param_scheduler = []

train_cfg = dict(
    type='RDPPTrainLoop',
    max_epochs=300,
    category_epochs=rdpp_category_epochs,
    val_begin=1,
    val_interval=1,
)
val_cfg = dict(type='ADValLoop')
test_cfg = dict(type='ADTestLoop')

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3, save_last=True),
)

randomness = dict(seed=111, deterministic=True)

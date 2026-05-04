"""MemSeg strict official-compatible config for MVTec AD.

Reference: TooTouch/MemSeg (`configs.yaml`)

Frozen reference protocol:
- resize `288x288` then center crop to `256x256`
- batch_size = 8
- num_workers = 0
- 5000 total training steps
- AdamW(lr=0.003, weight_decay=5e-4)
- warmup ratio = 0.1, min_lr = 1e-4
- freeze only `layer1/2/3` of the ResNet18 feature extractor
- memory bank built once from 30 randomly shuffled train samples
- alternating normal / synthetic-anomaly sampling during training
"""

_base_ = [
    '../_base_/default_runtime.py',
]

benchmark_multi_class = False
benchmark_result_selector = dict(
    mode='best_balanced',
    metrics=['image_auroc', 'pixel_auroc', 'aupro'],
    tie_break_metric='image_ap',
)

data_root = 'data/mvtec_ad'
resize_size = 288
img_size = 256

backbone = dict(
    type='TIMMBackbone',
    model_name='resnet18',
    pretrained=True,
    features_only=True,
    out_indices=(0, 1, 2, 3, 4),
    frozen=False,
    frozen_names=('layer1', 'layer2', 'layer3'),
    frozen_names_eval=False,
)

model = dict(
    type='MemSegDetector',
    backbone=backbone,
    nb_memory_sample=30,
    dtd_path='auto',
    l1_weight=0.6,
    focal_weight=0.4,
    focal_gamma=4,
    anomaly_ratio=0.5,
    alternate_anomaly_sampling=True,
    require_texture_source=True,
    memory_bank_seed=42,
    use_imgaug=True,
    anomaly_source_resize=resize_size,
    anomaly_source_crop=img_size,
)

train_pipeline = [
    dict(type='LoadImage'),
    dict(type='LoadMask'),
    dict(type='ResizeAD', size=resize_size),
    dict(type='CenterCrop', size=img_size),
    dict(type='NormalizeAD'),
    dict(type='PackADInputs'),
]

test_pipeline = [
    dict(type='LoadImage'),
    dict(type='LoadMask'),
    dict(type='ResizeAD', size=resize_size),
    dict(type='CenterCrop', size=img_size),
    dict(type='NormalizeAD'),
    dict(type='PackADInputs'),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='PersistentShuffleSampler', shuffle=True, seed=42, round_up=False),
    dataset=dict(
        type='MVTecADDataset',
        data_root=data_root,
        split='train',
        multi_class=False,
        cls_names=['bottle'],
        pipeline=train_pipeline,
    ),
)

test_dataloader = dict(
    batch_size=8,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MVTecADDataset',
        data_root=data_root,
        split='test',
        multi_class=False,
        cls_names=['bottle'],
        pipeline=test_pipeline,
    ),
)

val_dataloader = test_dataloader

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.003,
        weight_decay=0.0005,
    ),
)

# Official scheduler initializes lr at `min_lr`, linearly warms up to `lr`,
# then follows a single cosine cycle back to `min_lr`.
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 30.0,  # 1e-4 / 3e-3
        by_epoch=False,
        begin=0,
        end=500,
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=4500,
        eta_min=1e-4,
        by_epoch=False,
        begin=500,
        end=5000,
    ),
]

train_cfg = dict(
    by_epoch=False,
    max_iters=5000,
    val_interval=100,
)
val_cfg = dict()
test_cfg = dict(type='ADTestLoop')

test_evaluator = dict(type='AnomalyDetectionMetric')
val_evaluator = test_evaluator

custom_hooks = [
    dict(type='MemSegStrictTrainHook'),
    dict(type='MemoryBankHook'),
]

randomness = dict(seed=42, deterministic=False)

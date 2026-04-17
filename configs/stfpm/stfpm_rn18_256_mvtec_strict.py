"""STFPM strict official-alignment config.

Reference freeze:
- Official repository: `gdwang08/STFPM`
- Commit: `2598a5e35fd02f2f9dcfd0f3e8249adc22320e59`
- Runtime authority: `main.py` / `evaluate.py`
- README result table is auxiliary evidence only
"""

_base_ = [
    '../_base_/default_runtime.py',
]

data_root = 'data/mvtec_ad'
img_size = 256

benchmark_multi_class = False
benchmark_preserve_checkpoint_hooks = True
benchmark_resume_existing = True
benchmark_test_after_train = True
benchmark_checkpoint_source = 'best'
benchmark_result_selector = dict(mode='last')
benchmark_timeout = 14400

randomness = dict(seed=0, deterministic=False)

train_pipeline = [
    dict(type='LoadImage'),
    dict(type='LoadMask'),
    dict(type='ResizeAD', size=img_size),
    dict(type='NormalizeAD'),
    dict(type='PackADInputs'),
]

test_pipeline = [
    dict(type='LoadImage'),
    dict(type='LoadMask'),
    dict(type='ResizeAD', size=img_size),
    dict(type='NormalizeAD'),
    dict(type='PackADInputs'),
]

train_dataloader = dict(
    batch_size=32,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='MVTecADDataset',
        data_root=data_root,
        split='train',
        multi_class=False,
        train_val_split_ratio=0.2,
        train_val_split_seed=0,
        train_val_split_subset='train',
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MVTecADDataset',
        data_root=data_root,
        split='train',
        multi_class=False,
        train_val_split_ratio=0.2,
        train_val_split_seed=0,
        train_val_split_subset='val',
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
        multi_class=False,
        pipeline=test_pipeline,
    ),
)

val_evaluator = dict(type='AnomalyMapMeanMetric')
test_evaluator = dict(type='AnomalyDetectionMetric')

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1000000,
        max_keep_ckpts=1,
        save_last=False,
        save_best='ad/score_mean',
        rule='less',
    ),
)

model = dict(
    type='STFPMDetector',
    reference_impl='official',
    backbone=dict(
        type='FeatureExtractor',
        backbone_name='resnet18',
        pretrained=True,
        out_indices=(1, 2, 3),
        frozen=True,
    ),
)

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.4, momentum=0.9, weight_decay=1e-4),
)

param_scheduler = []

train_cfg = dict(by_epoch=True, max_epochs=100, val_begin=1, val_interval=1)
val_cfg = dict(type='ADValLoop')
test_cfg = dict(type='ADTestLoop')

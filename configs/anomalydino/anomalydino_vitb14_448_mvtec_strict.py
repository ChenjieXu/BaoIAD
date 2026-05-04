"""Strict AnomalyDINO config aligned to the official main branch.

Reference freeze:
- repo: dammsi/AnomalyDINO
- commit: b9d1c2648e3a5247437d4d953d907a8f3d994457
- setting: MVTec AD, 1-shot, preprocess=agnostic, seed=0
"""

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/mvtec_ad.py',
]

img_size = 448

model = dict(
    type='AnomalyDINODetector',
    backbone=dict(
        type='DINOv2Backbone',
        model_name='dinov2_vitb14',
        frozen=True,
    ),
    k=1,
    preprocess='agnostic',
    mask_ref_images=False,
    top_ratio=0.01,
    gaussian_sigma=4.0,
    few_shot=1,
    few_shot_seed=0,
)

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.0, momentum=0.0)
)
param_scheduler = []

train_cfg = dict(by_epoch=True, max_epochs=1, val_interval=1)
val_cfg = dict()
test_cfg = dict(type='ADTestLoop')

benchmark_multi_class = False

randomness = dict(seed=0, deterministic=False)

_strict_pipeline = [
    dict(type='LoadImage'),
    dict(type='LoadMask'),
    dict(type='ResizeAD', size=img_size, keep_ratio=True),
    dict(type='NormalizeAD'),
    dict(type='PackADInputs'),
]

train_dataloader = dict(
    batch_size=1,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        pipeline=_strict_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=1,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        pipeline=_strict_pipeline,
    ),
)
test_dataloader = dict(
    batch_size=1,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        pipeline=_strict_pipeline,
    ),
)

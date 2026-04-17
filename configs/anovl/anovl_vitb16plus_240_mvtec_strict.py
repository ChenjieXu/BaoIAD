"""AnoVL strict official-alignment config.

Reference freeze:
- Official repository: `hq-deng/AnoVL`
- Commit: `3a70bfdaea6baf1eeb140c5de8155b535bd94833`
- Runtime authority: `test_zero_shot.sh` / `vl_test.py`
- Official zero-shot protocol: `ViT-B-16-plus-240 + laion400m_e32`
"""

_base_ = [
    '../_base_/default_runtime.py',
]

data_root = 'data/mvtec_ad'
img_size = 240
openclip_local_pretrained = (
    'pretrained/open_clip/vit_b_16_plus_240-laion400m_e32-699c4b84.pt'
)

# AnoVL is an eval-time zero-shot method.
# `train.py` still needs a train loop for smoke coverage, but the only
# learnable optimization happens inside predict() as per-image TTA.
benchmark_eval_only = True
benchmark_multi_class = True

randomness = dict(seed=111, deterministic=False)

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
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='MVTecADDataset',
        data_root=data_root,
        split='train',
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

test_dataloader = dict(
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

val_evaluator = dict(type='AnomalyDetectionMetric')
test_evaluator = dict(type='AnomalyDetectionMetric')

model = dict(
    type='AnoVLDetector',
    clip_model='ViT-B-16-plus-240',
    pretrained=openclip_local_pretrained,
    class_name='object',
    image_size=240,
    features_list=[3, 6, 9, 12],
    tta_enabled=True,
    tta_epochs=5,
    tta_lr=1e-3,
    smoothing_kernel=3,
)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1, val_interval=1)
val_cfg = dict(type='ADValLoop')
test_cfg = dict(type='ADTestLoop')

optim_wrapper = dict(optimizer=dict(type='SGD', lr=0.0))
param_scheduler = []

_base_ = ['../_base_/default_runtime.py']

# Strict anomalib-aligned EfficientAD-S configuration.
benchmark_keep_dataloader_workers = True
benchmark_preserve_checkpoint_hooks = True
benchmark_resume_existing = True
benchmark_timeout = 21600
data_root = 'data/mvtec_ad'
img_size = 256

train_pipeline = [
    dict(type='LoadImage'),
    dict(type='LoadMask'),
    dict(type='ResizeAD', size=img_size),
    dict(type='NormalizeAD', mean=(0, 0, 0), std=(255, 255, 255)),
    dict(type='PackADInputs'),
]
test_pipeline = [
    dict(type='LoadImage'),
    dict(type='LoadMask'),
    dict(type='ResizeAD', size=img_size),
    dict(type='NormalizeAD', mean=(0, 0, 0), std=(255, 255, 255)),
    dict(type='PackADInputs'),
]

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='MVTecADDataset',
        data_root=data_root,
        split='train',
        pipeline=train_pipeline,
    ),
)
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MVTecADDataset',
        data_root=data_root,
        split='test',
        pipeline=test_pipeline,
    ),
)
val_dataloader = test_dataloader
test_evaluator = dict(type='AnomalyDetectionMetric')
val_evaluator = test_evaluator

model = dict(
    type='EfficientADDetector',
    pdn_variant='small',
    padding=False,
    pad_maps=True,
    teacher_pretrained='auto',
    data_preprocessor=dict(type='mmengine.model.ImgDataPreprocessor'),
)

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=1e-4, weight_decay=1e-5),
)

param_scheduler = [
    dict(type='StepLR', step_size=66500, gamma=0.1),
]

train_cfg = dict(by_epoch=False, max_iters=70000, val_interval=5000)
val_cfg = dict()
test_cfg = dict(type='ADTestLoop')

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=2500,
        save_last=True,
        max_keep_ckpts=2,
    ),
)

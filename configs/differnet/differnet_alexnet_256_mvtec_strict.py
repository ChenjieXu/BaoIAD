# DifferNet with AlexNet backbone, aligned with original paper (WACV 2021)
# Paper: "Same Same But DifferNet: Semi-Supervised Defect Detection with Normalizing Flows"
# Key: multi-scale input (448/224/112), 64 test-time rotations, MSE scoring
# Reference img_AUROC: 94.9% (MVTec AD) — image-level only method
_base_ = ['../_base_/default_runtime.py']

data_root = 'data/mvtec_ad'
img_size = 448  # Original DifferNet uses 448x448

# DifferNet uses multi-scale internally (resizes to 448/224/112)
# Pipeline handles initial resize and normalization
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

train_dataloader = dict(batch_size=24, num_workers=4, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(type='MVTecADDataset', data_root=data_root, split='train',
                 cls_names=['bottle'], pipeline=train_pipeline))
test_dataloader = dict(batch_size=16, num_workers=4, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='MVTecADDataset', data_root=data_root, split='test',
                 cls_names=['bottle'], pipeline=test_pipeline))
val_dataloader = test_dataloader
test_evaluator = dict(type='AnomalyDetectionMetric')
val_evaluator = test_evaluator
benchmark_result_selector = dict(mode='best', metric='image_auroc')

model = dict(
    type='DifferNetDetector',
    backbone='alexnet',
    data_preprocessor=dict(type='mmengine.model.ImgDataPreprocessor'),
    n_coupling_blocks=8,
    clamp=3.0,  # original paper's clamp_alpha
    multi_scale=True,
    n_transforms=64,
    n_train_transforms=4,
    colorjitter_brightness=0.0,  # original: no color jitter
    colorjitter_contrast=0.0,
    colorjitter_saturation=0.0,
    scales=((448, 448), (224, 224), (112, 112)),
)

# Original: Adam, lr=2e-4, betas=(0.8, 0.8), eps=1e-4, weight_decay=1e-5, constant LR, 192 epochs
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=2e-4, betas=(0.8, 0.8), eps=1e-4, weight_decay=1e-5),
)
param_scheduler = []
train_cfg = dict(by_epoch=True, max_epochs=192, val_interval=24)
val_cfg = dict()
test_cfg = dict(type='ADTestLoop')

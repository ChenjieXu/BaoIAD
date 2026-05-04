# DRAEM config aligned with ADer reference (0.940 img_AUROC)
# Key: NO ImageNet normalization — images in [0,1] range
# Augmentation moved to DRAEMDataset for stable training
_base_ = ['../_base_/default_runtime.py']

data_root = 'data/mvtec_ad'
img_size = 256

# Test pipeline uses standard transforms
test_pipeline = [
    dict(type='LoadImage'),
    dict(type='LoadMask'),
    dict(type='ResizeAD', size=img_size),
    dict(type='ScaleNormalizeAD'),
    dict(type='PackADInputs'),
]

# Category to train/test on (override via --cfg-options)
category = 'bottle'

# Training uses DRAEMDataset with augmentation in __getitem__
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='DRAEMDataset',
        data_root=data_root,
        cls_names=[category],
        dtd_path='auto',
        img_size=img_size,
        beta_range=(0.0, 0.8),  # ADer: beta = rand()*0.8 → [0, 0.8]
        anomaly_ratio=0.5,  # 50% chance of generating anomaly
        pipeline=[dict(type='PackDRAEMInputs')],
    )
)

test_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MVTecADDataset',
        data_root=data_root,
        split='test',
        cls_names=[category],
        pipeline=test_pipeline
    )
)
val_dataloader = test_dataloader
test_evaluator = dict(type='AnomalyDetectionMetric')
val_evaluator = test_evaluator

# Model no longer has augmentation parameters
model = dict(
    type='DRAEMDetector',
    ssim_weight=1.0,  # ADer uses lam=1.0 for all losses (MSE:SSIM:Focal = 1:1:1)
    data_preprocessor=dict(type='mmengine.model.ImgDataPreprocessor'),
)

# ADer 300e: Adam lr=1e-4, MultiStepLR at 80% and 90% epochs (240, 270), gamma=0.2
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=1e-4),
)
param_scheduler = [
    dict(type='MultiStepLR', milestones=[240, 270], gamma=0.2, by_epoch=True),
]
train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=1)
val_cfg = dict()
test_cfg = dict(type='ADTestLoop')

# Early stopping disabled for debugging
custom_hooks = [
    dict(type='MemoryBankHook'),
]

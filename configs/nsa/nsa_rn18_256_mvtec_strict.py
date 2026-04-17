# NSA (logistic) config aligned with the official MVTec setting
# (Schluter et al., ECCV 2022).
# Key alignment points:
# - ResNet-18 backbone (not WRN-50-2)
# - Sigmoid + BCE supervision on logistic-intensity labels
# - One synthetic anomaly per training sample (official self-supervised task)
# - Dataset-side source-image state (`prev_idx`) and patch synthesis
# - Skip background logic for object categories
# - Object-class evaluation uses 256 input -> CenterCrop(224) -> Pad(16)
# - 320 epochs (560 for hazelnut/metal_nut/screw), cosine annealing from 1e-3 to 1e-6
# - Logistic-intensity label generation with per-category parameters
# - No extra Gaussian smoothing at inference
_base_ = ['../_base_/default_runtime.py']

data_root = 'data/mvtec_ad'

train_pipeline = [
    dict(type='PackADInputs'),
]

# Test transforms using category-aware NSATestTransform
# Objects: 224x224, Textures: 256x256
test_pipeline = [
    dict(type='LoadImage'),
    dict(type='LoadMask'),
    dict(type='ResizeAD', size=256),  # Initial resize
    dict(type='NSATestTransform'),    # Category-aware test transform
    dict(type='ScaleNormalizeAD'),
    dict(type='PackADInputs'),
]

train_dataloader = dict(batch_size=64, num_workers=8, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='NSATrainDataset',
        data_root=data_root,
        cls_names=['bottle'],
        anomaly_ratio=1.0,
        use_logistic_labels=True,
        pipeline=train_pipeline,
    ))
test_dataloader = dict(batch_size=32, num_workers=4, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='MVTecADDataset', data_root=data_root, split='test',
                 cls_names=['bottle'], pipeline=test_pipeline))
val_dataloader = test_dataloader
test_evaluator = dict(type='AnomalyDetectionMetric')
val_evaluator = test_evaluator

model = dict(
    type='NSADetector',
    backbone=dict(type='RawBackbone', backbone_name='resnet18'),
    anomaly_ratio=1.0,
    seg_base_width=64,
    buffer_size=1000,
    gaussian_sigma=0.0,
    data_preprocessor=dict(type='mmengine.model.ImgDataPreprocessor'),
    # Official MVTec reference uses logistic-intensity labels.
    use_logistic_labels=True,
    median_filter_radius=5,
)

# Original paper: Adam, cosine annealing 1e-3 → 1e-6 over 320 epochs
# Note: Some categories (hazelnut, metal_nut, screw) need 560 epochs
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=1e-3),
)
param_scheduler = [
    dict(type='CosineAnnealingLR', T_max=320, eta_min=1e-6, by_epoch=True),
]
train_cfg = dict(by_epoch=True, max_epochs=320, val_interval=50)
val_cfg = dict()
test_cfg = dict(type='ADTestLoop')


# DSR config aligned with anomalib
# Key: NO ImageNet normalization — images in [0,1] range
# Phases 2 & 3 auto-switched via upsampling_train_ratio
_base_ = ['../_base_/default_runtime.py']

data_root = 'data/mvtec_ad'
img_size = 256

# Custom pipeline: ScaleNormalizeAD for [0,1] images (no ImageNet norm)
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='LoadMask'),
    dict(type='ResizeAD', size=img_size),
    dict(type='ScaleNormalizeAD'),
    dict(type='PackADInputs'),
]
test_pipeline = [
    dict(type='LoadImage'),
    dict(type='LoadMask'),
    dict(type='ResizeAD', size=img_size),
    dict(type='ScaleNormalizeAD'),
    dict(type='PackADInputs'),
]

train_dataloader = dict(batch_size=4, num_workers=4, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(type='MVTecADDataset', data_root=data_root, split='train',
                 cls_names=['bottle'], pipeline=train_pipeline))
test_dataloader = dict(batch_size=4, num_workers=4, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='MVTecADDataset', data_root=data_root, split='test',
                 cls_names=['bottle'], pipeline=test_pipeline))
val_dataloader = test_dataloader
test_evaluator = dict(type='AnomalyDetectionMetric')
val_evaluator = test_evaluator

model = dict(
    type='DSRDetector',
    num_hiddens=128,
    num_residual_layers=2,
    num_residual_hiddens=64,
    num_embeddings=4096,
    embedding_dim=128,
    phase=2,
    upsampling_train_ratio=0.7,  # switch to phase 3 at 70% of training
    pretrained_vqvae_path='auto',
    recon_loss=dict(type='MSELoss'),
    feat_loss=dict(type='MSELoss'),
    seg_loss=dict(type='FocalLoss', alpha=1.0, gamma=2.0),
    data_preprocessor=dict(type='mmengine.model.ImgDataPreprocessor'),
)

# Anomalib: Adam lr=0.0002; StepLR at 0.8*phase2_epochs
# With 100 epochs, phase2=70, anneal=56
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=2e-4, weight_decay=0),
)
param_scheduler = [
    dict(type='StepLR', step_size=56, gamma=0.1, by_epoch=True),
]
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=20)
val_cfg = dict()
test_cfg = dict(type='ADTestLoop')

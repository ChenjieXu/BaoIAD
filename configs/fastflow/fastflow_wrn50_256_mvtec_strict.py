_base_ = ['../_base_/default_runtime.py', '../_base_/datasets/mvtec_ad.py']
data_root = 'data/mvtec_ad'
train_dataloader = dict(batch_size=32, num_workers=4, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(type='MVTecADDataset', data_root=data_root, split='train', cls_names=['bottle'], pipeline={{_base_.train_pipeline}}))
test_dataloader = dict(batch_size=32, num_workers=4, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='MVTecADDataset', data_root=data_root, split='test', cls_names=['bottle'], pipeline={{_base_.test_pipeline}}))
val_dataloader = test_dataloader
test_evaluator = dict(type='AnomalyDetectionMetric')
val_evaluator = test_evaluator
model = dict(type='FastFlowDetector',
    backbone=dict(
        type='TIMMBackbone',
        model_name='wide_resnet50_2',
        features_only=True,
        out_indices=(1, 2, 3),
        frozen=True,
    ),
    flow_steps=8,
    conv3x3_only=False,
    hidden_ratio=1.0,
    clamp=2.0,
    input_size=(256, 256),
    data_preprocessor=dict(type='mmengine.model.ImgDataPreprocessor'),)
# anomalib: constant lr=0.001, no scheduler
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=0.001, weight_decay=0.00001),
)
param_scheduler = []
# anomalib: max_epochs=500 with early stopping (patience=3 on pixel_AUROC)
# BaoIAD: train 500 epochs, validate every 50
# Note: Early stopping not critical for alignment, subnet conv fix is more important
train_cfg = dict(by_epoch=True, max_epochs=500, val_interval=50)
val_cfg = dict()
test_cfg = dict(type='ADTestLoop')

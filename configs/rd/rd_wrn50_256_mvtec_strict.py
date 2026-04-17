# RD strict official-alignment config.
# Reference: RD4AD official repository, commit 6554076872c65f8784f6ece8cfb39ce77e1aee12.
_base_ = ['../_base_/default_runtime.py', '../_base_/datasets/mvtec_ad.py']

data_root = 'data/mvtec_ad'
benchmark_multi_class = False
benchmark_result_selector = dict(mode='last')
benchmark_keep_dataloader_workers = True

randomness = dict(seed=111, deterministic=True)

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='MVTecADDataset',
        data_root=data_root,
        split='train',
        multi_class=True,
        pipeline={{_base_.train_pipeline}},
    ),
)
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MVTecADDataset',
        data_root=data_root,
        split='test',
        multi_class=True,
        pipeline={{_base_.test_pipeline}},
    ),
)
val_dataloader = test_dataloader
test_evaluator = dict(type='AnomalyDetectionMetric')
val_evaluator = test_evaluator

model = dict(
    type='ReverseDistillation',
    backbone=dict(
        type='FeatureExtractor',
        backbone_name='wide_resnet50_2',
        pretrained=True,
        out_indices=(1, 2, 3),
        frozen=True,
    ),
    anomaly_map_mode='add',
    smooth_sigma=4.0,
    smoothing_backend='scipy',
)

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=0.005, betas=(0.5, 0.999), weight_decay=0.0),
)
param_scheduler = []
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=200, val_begin=10, val_interval=10)
val_cfg = dict(type='ADValLoop')
test_cfg = dict(type='ADTestLoop')

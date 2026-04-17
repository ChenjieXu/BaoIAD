_base_ = ['../_base_/default_runtime.py']

data_root = 'data/mvtec_ad'
img_size = 256
category = 'bottle'
norm_mean = (127.5, 127.5, 127.5)
norm_std = (127.5, 127.5, 127.5)

train_pipeline = [
    dict(type='LoadImage'),
    dict(type='LoadMask'),
    dict(type='ResizeAD', size=img_size),
    dict(type='NormalizeAD', mean=norm_mean, std=norm_std),
    dict(type='PackADInputs'),
]

test_pipeline = [
    dict(type='LoadImage'),
    dict(type='LoadMask'),
    dict(type='ResizeAD', size=img_size),
    dict(type='NormalizeAD', mean=norm_mean, std=norm_std),
    dict(type='PackADInputs'),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='MVTecADDataset',
        data_root=data_root,
        split='train',
        cls_names=[category],
        pipeline=train_pipeline,
    ),
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
        pipeline=test_pipeline,
    ),
)

val_dataloader = test_dataloader

test_evaluator = dict(type='AnomalyDetectionMetric')
val_evaluator = test_evaluator

model = dict(
    type='MemAEDetector',
    in_channels=3,
    frame_num=16,
    clip_mode='repeat_image',
    mem_dim=2000,
    shrink_thres=0.0025,
    entropy_loss_weight=0.0002,
    temporal_reduce_mode='mean',
    image_score_mode='spatiotemporal_mean',
    loss=dict(type='MSELoss'),
    data_preprocessor=dict(type='mmengine.model.ImgDataPreprocessor'),
)

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=1e-4, weight_decay=0),
)

param_scheduler = []

train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=10)
val_cfg = dict()
test_cfg = dict(type='ADTestLoop')

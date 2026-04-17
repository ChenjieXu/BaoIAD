data_root = 'data/mvtec_3d_ad'
img_size = 256

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
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='MVTec3DDataset',
        data_root=data_root,
        split='train',
        multi_class=True,
        pipeline=train_pipeline,
    ),
)

test_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MVTec3DDataset',
        data_root=data_root,
        split='test',
        multi_class=True,
        pipeline=test_pipeline,
    ),
)

val_dataloader = test_dataloader

test_evaluator = dict(type='AnomalyDetectionMetric')
val_evaluator = test_evaluator

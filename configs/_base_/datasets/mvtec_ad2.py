resolve_data_root = __import__('baoiad.paths', fromlist=['resolve_data_root']).resolve_data_root

data_root = str(resolve_data_root('mvtec_ad_2'))
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
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='MVTecAD2Dataset',
        data_root=data_root,
        split='train',
        multi_class=True,
        test_type='public',
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MVTecAD2Dataset',
        data_root=data_root,
        split='val',
        multi_class=True,
        test_type='public',
        pipeline=test_pipeline,
    ),
)

test_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MVTecAD2Dataset',
        data_root=data_root,
        split='test',
        multi_class=True,
        test_type='public',
        pipeline=test_pipeline,
    ),
)

test_evaluator = dict(type='AnomalyDetectionMetric')
val_evaluator = test_evaluator

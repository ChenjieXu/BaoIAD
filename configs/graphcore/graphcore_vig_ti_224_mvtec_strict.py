_base_ = [
    '../_base_/default_runtime.py',
]

custom_imports = dict(
    imports=['baoiad.datasets.samplers'],
    allow_failed_imports=False,
)

# Strict GraphCore mainline stays on a single global raw-max image score.
# Diagnose-only score overrides or alternative coreset starts belong in tools/*.
benchmark_multi_class = False
randomness = dict(seed=66, deterministic=False)

data_root = 'data/mvtec_ad'
img_size = 224

train_pipeline = [
    dict(type='LoadImage'),
    dict(type='LoadMask'),
    dict(type='GraphCorePreprocessAD', size=img_size, crop_size=img_size),
    dict(type='PackADInputs'),
]

test_pipeline = [
    dict(type='LoadImage'),
    dict(type='LoadMask'),
    dict(type='GraphCorePreprocessAD', size=img_size, crop_size=img_size),
    dict(type='PackADInputs'),
]

train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='OpenIADSubsetRandomSampler', shuffle=True, seed=66, round_up=False),
    dataset=dict(
        type='MVTecADDataset',
        data_root=data_root,
        split='train',
        multi_class=True,
        pipeline=train_pipeline,
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
        pipeline=test_pipeline,
    ),
)

val_dataloader = test_dataloader

model = dict(
    type='GraphCoreDetector',
    backbone=dict(
        type='GraphCoreViGBackbone',
        model_name='vig_ti_224_gelu',
        pretrained=True,
        checkpoint_path='pretrained/graphcore',
        frozen=True,
    ),
    n_neighbours=9,
    sampler_percentage=0.001,
    layer_num_1=3,
    layer_num_2=4,
    local_smoothing=False,
    input_size=(224, 224),
    smoothing_sigma=4.0,
    image_score_mode='raw_max',
    image_score_mode_overrides={},
    random_seed=66,
    coreset_initial_index=0,
)

custom_hooks = [dict(type='MemoryBankHook')]

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=1e-3, weight_decay=1e-5),
)

param_scheduler = []

train_cfg = dict(by_epoch=True, max_epochs=1, val_interval=1)
val_cfg = dict()
test_cfg = dict(type='ADTestLoop')

test_evaluator = dict(type='AnomalyDetectionMetric')
val_evaluator = test_evaluator

# Strict official CFlow alignment path.
#
# Reference:
# - repo: `.refs/cflow-ad`
# - commit: `b2ebf9e673a0aa46992a3b18367ec066a57bba89`
# - README MVTec commands use per-category input sizes instead of a fixed 256.

_base_ = [
    '../_base_/default_runtime.py',
]

data_root = 'data/mvtec_ad'

cflow_input_size_map = dict(
    bottle=512,
    cable=256,
    capsule=256,
    carpet=512,
    grid=512,
    hazelnut=256,
    leather=512,
    metal_nut=256,
    pill=256,
    screw=512,
    tile=512,
    toothbrush=512,
    transistor=128,
    wood=512,
    zipper=512,
)

train_pipeline = [
    dict(type='LoadImage'),
    dict(type='LoadMask'),
    dict(
        type='CFlowOfficialTransform',
        size_map=cflow_input_size_map,
        default_size=256,
        train=True,
    ),
    dict(type='NormalizeAD'),
    dict(type='PackADInputs'),
]

test_pipeline = [
    dict(type='LoadImage'),
    dict(type='LoadMask'),
    dict(
        type='CFlowOfficialTransform',
        size_map=cflow_input_size_map,
        default_size=256,
        train=False,
    ),
    dict(type='NormalizeAD'),
    dict(type='PackADInputs'),
]

train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='MVTecADDataset',
        data_root=data_root,
        split='train',
        cls_names=['bottle'],
        multi_class=False,
        pipeline=train_pipeline,
    ),
)

test_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MVTecADDataset',
        data_root=data_root,
        split='test',
        cls_names=['bottle'],
        multi_class=False,
        pipeline=test_pipeline,
    ),
)

val_dataloader = test_dataloader

model = dict(
    type='CFlowDetector',
    backbone=dict(
        type='FeatureExtractor',
        backbone_name='wide_resnet50_2',
        pretrained=True,
        out_indices=(2, 3, 4),
        frozen=True,
    ),
    coupling_blocks=8,
    clamp_alpha=1.9,
    condition_dim=128,
    permute_soft=True,
    fiber_batch_size=256,
    reference_repo='.refs/cflow-ad',
    require_official_reference=True,
)

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=2e-4, weight_decay=0.0),
)

param_scheduler = []

train_cfg = dict(
    type='CFlowOfficialTrainLoop',
    max_epochs=25,
    val_begin=1,
    val_interval=1,
    sub_epochs=8,
    lr_decay_rate=0.1,
    lr_warm=True,
    lr_warm_epochs=2,
    lr_cosine=True,
    warmup_ratio=0.1,
)
val_cfg = dict(type='ADValLoop')
test_cfg = dict(type='ADTestLoop')

val_evaluator = dict(type='AnomalyDetectionMetric')
test_evaluator = val_evaluator

benchmark_result_selector = dict(
    mode='best_per_metric',
    metrics=['image_auroc', 'pixel_auroc', 'aupro'],
)
benchmark_keep_dataloader_workers = True

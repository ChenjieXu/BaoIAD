"""ResAD BaoIAD sidecar config with official detector internals.

Reference freeze:
- Official repository: `xcyao00/ResAD`
- Commit: `0a9b35e421e3f802b5bcf8e15378e2b4ebdba8c9`
- Runtime authority: `main.py`, `train.py`, `validate.py`, `losses/loss.py`
- Naming note: this file keeps the historical `*_256_*` compatibility name,
  but the frozen official preprocessing is `Resize(224) + CenterCrop(224)`.
- Mainline note: the strict final protocol identity now lives in
  `configs/resad/resad_official_visa_to_mvtec.py` and
  `tools/resad_official_eval.py`.
"""

_base_ = [
    '../_base_/default_runtime.py',
]

data_root = 'data/mvtec_ad'
img_size = 224
ref_feature_dir = 'pretrained/resad/ref_features/w50/mvtec_4shot'

# This sidecar keeps the official detector / preprocessing semantics inside the
# MMEngine runner, but it still trains/tests on MVTec. The official strict
# protocol is VisA -> MVTec and is archived separately via
# `tools/resad_official_eval.py`.
benchmark_multi_class = True

randomness = dict(seed=42, deterministic=False)

train_pipeline = [
    dict(type='LoadImage'),
    dict(type='LoadMask'),
    dict(type='ResizeAD', size=img_size, keep_ratio=True),
    dict(type='CenterCrop', size=img_size),
    dict(type='NormalizeAD'),
    dict(type='PackADInputs'),
]

test_pipeline = [
    dict(type='LoadImage'),
    dict(type='LoadMask'),
    dict(type='ResizeAD', size=img_size, keep_ratio=True),
    dict(type='CenterCrop', size=img_size),
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
        multi_class=True,
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
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

val_evaluator = dict(type='AnomalyDetectionMetric')
test_evaluator = dict(type='AnomalyDetectionMetric')

model = dict(
    type='ResADDetector',
    backbone=dict(
        type='TIMMBackbone',
        model_name='wide_resnet50_2',
        features_only=True,
        out_indices=(1, 2, 3),
        pretrained=True,
        frozen=True,
    ),
    n_shot=4,
    num_embeddings=1536,
    pos_embed_dim=256,
    coupling_layers=10,
    clamp_alpha=1.9,
    fdm_alpha=0.4,
    r_max=0.4,
    occ_lambda=1.0,
    flow_lambda=1.0,
    first_stage_epochs=10,
    smooth_sigma=4.0,
    margin_tau=0.1,
    bgspp_lambda=1.0,
    pos_beta=0.05,
    ref_feature_dir=ref_feature_dir,
    num_ref_shot=4,
    total_ref_shot=4,
    data_root=data_root,
    input_size=img_size,
    strict_ref_features=True,
)

optim_wrapper = dict(
    constructor='baoiad.ResADOptimWrapperConstructor',
    vq=dict(
        optimizer=dict(type='Adam', lr=1e-5, weight_decay=5e-4),
    ),
    constraintor=dict(
        optimizer=dict(type='Adam', lr=1e-5, weight_decay=5e-4),
    ),
    flow=dict(
        optimizer=dict(type='Adam', lr=1e-5, weight_decay=5e-4),
    ),
)

param_scheduler = [
    dict(type='MultiStepLR', milestones=[70, 90], gamma=0.1, by_epoch=True),
]

# Official ResAD trains VQ -> constraintor -> flow each iteration, with the
# first 10 epochs using only the normal flow branch.
train_cfg = dict(
    type='ResADOfficialTrainLoop',
    max_epochs=100,
    val_interval=10,
    first_stage_epochs=10,
    N_batch=8192,
)
val_cfg = dict(type='ADValLoop')
test_cfg = dict(type='ADTestLoop')

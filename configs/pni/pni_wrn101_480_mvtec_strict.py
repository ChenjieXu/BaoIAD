"""PNI strict official-alignment config.

Reference freeze:
- Official repository: `wogur110/PNI_anomaly_detection`
- Commit: `958626a3408fc3b1561751abfd1ce534f66603f6`
- Runtime authority: `train_coreset_distribution.py`
"""

_base_ = [
    '../_base_/default_runtime.py',
]

data_root = 'data/mvtec_ad'
resize_size = 512
crop_size = 480

benchmark_multi_class = False
randomness = dict(seed=23, deterministic=False)

train_pipeline = [
    dict(type='LoadImage'),
    dict(type='LoadMask'),
    dict(type='ResizeAD', size=resize_size, keep_ratio=True),
    dict(type='CenterCrop', size=crop_size),
    dict(type='NormalizeAD'),
    dict(type='PackADInputs'),
]

test_pipeline = [
    dict(type='LoadImage'),
    dict(type='LoadMask'),
    dict(type='ResizeAD', size=resize_size, keep_ratio=True),
    dict(type='CenterCrop', size=crop_size),
    dict(type='NormalizeAD'),
    dict(type='PackADInputs'),
]

train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='MVTecADDataset',
        data_root=data_root,
        split='train',
        multi_class=False,
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MVTecADDataset',
        data_root=data_root,
        split='test',
        multi_class=False,
        pipeline=test_pipeline,
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MVTecADDataset',
        data_root=data_root,
        split='test',
        multi_class=False,
        pipeline=test_pipeline,
    ),
)

val_evaluator = dict(type='AnomalyDetectionMetric')
test_evaluator = dict(type='AnomalyDetectionMetric')

model = dict(
    type='PNI',
    backbone=dict(
        type='TIMMBackbone',
        model_name='wide_resnet101_2',
        pretrained=False,
        checkpoint_path='pretrained/wide_resnet101_2-32ee1156.pth',
        features_only=True,
        out_indices=(2, 3),
        frozen=True,
    ),
    head=dict(
        type='PNIHead',
        coreset_ratio=0.01,
        distribution_size=2048,
        neighborhood_size=9,
        mlp_layers=10,
        mlp_channels=2048,
        temperature=2.0,
        lambda_param=1.0,
        num_neighbors=3,
        input_size=(crop_size, crop_size),
        mlp_epochs=15,
        mlp_lr=1e-3,
        prob_gamma=0.99,
        softmax_nb_gamma=0.5,
        softmax_coor_gamma=0.5,
        blur_sigma=8.0,
        mlp_batch_size=2048,
        max_train_samples=0,
        candidate_neighbors=100,
        patchsize=5,
        patchstride=1,
        pretrain_embed_dimension=1024,
        target_embed_dimension=1024,
        approximate_coreset=False,
        mlp_val_ratio=0.1,
        coreset_prefilter_size=12000,
        coreset_projection_dim=128,
        search_chunk_size=1024,
        # Keep exact-official math but avoid falling back to CPU during
        # full-bottle coordinate histogram construction.
        histogram_chunk_size=16,
        log_predict_stats=False,
    ),
    # Official protocol: freeze ImageNet-pretrained WRN101 and fit memory banks
    # plus the neighborhood MLP on single-class MVTec categories.
    freeze_backbone=True,
)

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=1e-3, weight_decay=0.0),
)

param_scheduler = []

train_cfg = dict(by_epoch=True, max_epochs=1, val_interval=1)
val_cfg = dict()
test_cfg = dict(type='ADTestLoop')

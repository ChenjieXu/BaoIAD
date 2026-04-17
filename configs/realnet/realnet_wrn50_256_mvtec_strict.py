# RealNet: Feature Reconstruction Network for Anomaly Detection (CVPR 2024)
# Reference config: .refs/RealNet_ref/experiments/MVTec-AD/realnet.yaml

_base_ = ['../_base_/default_runtime.py']

data_root = 'data/mvtec_ad'
img_size = 256
category = 'bottle'

test_pipeline = [
    dict(type='LoadImage'),
    dict(type='LoadMask'),
    dict(type='ResizeAD', size=img_size),
    dict(type='NormalizeAD'),
    dict(type='PackADInputs'),
]

train_dataloader = dict(
    batch_size=16,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RealNetTrainDataset',
        data_root=data_root,
        cls_names=[category],
        img_size=img_size,
        dataset_type='mvtec',
        dtd_dir='auto',
        sdas_dir='auto',
        dtd_transparency_range=(0.2, 1.0),
        sdas_transparency_range=(0.5, 1.0),
        perlin_scale=6,
        min_perlin_scale=0,
        anomaly_types=dict(
            normal=0.5,
            sdas=0.5,
        ),
        pipeline=[dict(type='PackRealNetInputs')],
    ),
)

test_dataloader = dict(
    batch_size=16,
    num_workers=0,
    persistent_workers=False,
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
# Official RealNet uses softmax channel 1 as anomaly score and relies on
# evaluator-side AUROC flipping for polarity diagnostics when needed.
test_evaluator = dict(
    type='AnomalyDetectionMetric',
    flip_auroc_if_below_half=True,
)
val_evaluator = test_evaluator

model = dict(
    type='RealNetDetector',
    backbone=dict(
        type='TIMMBackbone',
        model_name='wide_resnet50_2',
        pretrained=True,
        features_only=True,
        out_indices=(1, 2, 3, 4),
        frozen=True,
    ),
    structure=[
        dict(name='block1', layers=[dict(idx='layer1', planes=256)], stride=4),
        dict(name='block2', layers=[dict(idx='layer2', planes=512)], stride=8),
        dict(name='block3', layers=[dict(idx='layer3', planes=512)], stride=16),
        dict(name='block4', layers=[dict(idx='layer4', planes=256)], stride=32),
    ],
    init_bsn=64,
    reconstruction_type='official',
    num_res_blocks=2,
    hide_channels_ratio=0.5,
    channel_mult=[1, 2, 4],
    attention_mult=[2, 4],
    num_residual_layers=2,
    rrs_modes=['max', 'mean'],
    rrs_mode_numbers=[256, 256],
    image_score_pool_size=(16, 16),
    anomaly_channel_index=1,
    predict_invert_map=False,
    seg_loss=dict(type='CrossEntropyLoss'),
    feat_loss=dict(type='MSELoss'),
)

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=1e-4, betas=(0.9, 0.999)),
)
# Official RealNet MVTec config uses random_seed=100.
randomness = dict(seed=100, deterministic=False)
train_cfg = dict(by_epoch=True, max_epochs=1000, val_interval=5)
val_cfg = dict()
test_cfg = dict(type='ADTestLoop')

custom_hooks = [dict(type='RealNetInitHook')]

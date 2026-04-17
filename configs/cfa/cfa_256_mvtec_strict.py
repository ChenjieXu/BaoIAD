_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/mvtec_ad.py',
    '../_base_/schedules/schedule_100e.py',
]

model = dict(
    type='CFADetector',
    backbone=dict(
        type='FeatureExtractor',
        backbone_name='wide_resnet50_2',
        pretrained=True,
        out_indices=(1, 2, 3),
        frozen=True,
    ),
    gamma_c=1,
    gamma_d=1,
    num_nearest_neighbors=3,
    num_hard_negative_features=3,
    radius=1e-5,
    num_init_batches=3,
    sigma=4,
)

# Override optimizer to match anomalib: AdamW with amsgrad
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=1e-3, weight_decay=5e-4, amsgrad=True),
)

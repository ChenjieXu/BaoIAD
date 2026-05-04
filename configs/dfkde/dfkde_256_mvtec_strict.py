_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/mvtec_ad.py',
]

model = dict(
    type='DFKDEDetector',
    backbone=dict(
        type='TIMMBackbone',
        model_name='resnet18',
        pretrained=True,
        features_only=True,
        out_indices=(4,),
        frozen=True,
    ),
    n_pca_components=16,
    feature_scaling_method='scale',
    max_training_points=40000,
)

# DFKDE is memory-bank based, no real training needed
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=1e-3, weight_decay=1e-5),
)

param_scheduler = []

train_cfg = dict(by_epoch=True, max_epochs=1, val_interval=1)
val_cfg = dict()
test_cfg = dict(type='ADTestLoop')

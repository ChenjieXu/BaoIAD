# Canonical UniAD strict config: ADer-aligned MUAD training on all 15 MVTec
# categories. The `wrn50` filename is retained only as a compatibility alias;
# the actual strict backbone is EfficientNet-B4 with 4 feature stages.
# ADer reference: 92.5% image AUROC (MUAD 100ep)
_base_ = ['../_base_/default_runtime.py', '../_base_/datasets/mvtec_ad.py']

data_root = 'data/mvtec_ad'
benchmark_multi_class = True
benchmark_keep_dataloader_workers = True
benchmark_preserve_checkpoint_hooks = True

# multi_class=True loads all 15 categories together (~3629 training images)
train_dataloader = dict(batch_size=8, num_workers=4, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(type='MVTecADDataset', data_root=data_root, split='train',
                 multi_class=True, pipeline={{_base_.train_pipeline}}))
test_dataloader = dict(batch_size=8, num_workers=4, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='MVTecADDataset', data_root=data_root, split='test',
                 multi_class=True, pipeline={{_base_.test_pipeline}}))
val_dataloader = test_dataloader
test_evaluator = dict(type='AnomalyDetectionMetric')
val_evaluator = test_evaluator

model = dict(
    type='UniADDetector',
    backbone=dict(
        type='TIMMBackbone',
        model_name='tf_efficientnet_b4',
        features_only=True,
        out_indices=(0, 1, 2, 3),
        pretrained=True,
        frozen=True,
    ),
    feature_size=(16, 16),
    neighbor_size=(8, 8),
    image_score_mode='pooled_topk_mean',
    image_score_topk=128,
    activation='relu',
    normalize_before=False,
    neighbor_mask_layers=(True, True, True),
    use_neighbor_mask=True,
    loss=dict(type='MSELoss'),
    data_preprocessor=dict(type='mmengine.model.ImgDataPreprocessor'),
)

# ADer uses lr=1e-4 * batch_size / 8 with AdamW on EfficientNet-B4 features
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=1e-4, betas=(0.9, 0.999)),
)
param_scheduler = [
    dict(type='StepLR', step_size=80, gamma=0.1, by_epoch=True),
]
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=10)
val_cfg = dict()
test_cfg = dict(type='ADTestLoop')

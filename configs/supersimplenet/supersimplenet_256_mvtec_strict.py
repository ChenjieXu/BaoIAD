_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/mvtec_ad.py',
    '../_base_/backbones/wide_resnet50_raw.py',
]

# Strict anomalib-aligned SuperSimpleNet config.
# Reference: `.refs/anomalib/src/anomalib/models/image/supersimplenet/`
# - backbone: `wide_resnet50_2.tv_in1k`
# - layers: `layer2`, `layer3`
# - adaptor lr=1e-4, weight_decay=1e-2
# - segdec lr=2e-4, weight_decay=1e-5
# - official anomalib README recommends 300 epochs for stable training
# - MultiStepLR milestones remain at 80% / 90% of the training budget, gamma=0.4

benchmark_result_selector = dict(mode='last')
benchmark_rescale_epoch_schedulers = True

model = dict(
    type='SuperSimpleNetDetector',
    backbone={{_base_.backbone}},
    data_preprocessor=dict(type='mmengine.model.ImgDataPreprocessor'),
    layers=['layer2', 'layer3'],
    perlin_threshold=0.2,
    stop_grad=True,
    adapt_cls_features=False,
)

optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.00001),
    paramwise_cfg=dict(
        custom_keys={
            'adaptor': dict(lr_mult=0.5, decay_mult=1000.0),
        }
    ),
)

param_scheduler = [
    dict(type='MultiStepLR', milestones=[240, 270], gamma=0.4, by_epoch=True),
]

train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=10)
val_cfg = dict()
test_cfg = dict(type='ADTestLoop')
test_evaluator = dict(type='AnomalyDetectionMetric', image_score_field='pred_score_max')
val_evaluator = test_evaluator

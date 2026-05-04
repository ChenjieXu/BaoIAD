_base_ = [
    '../_base_/default_runtime.py',
]

benchmark_train_script = 'tools/train_ast.py'
benchmark_preserve_checkpoint_hooks = True
benchmark_keep_dataloader_workers = True
benchmark_multi_class = False
benchmark_result_selector = dict(mode='last')

img_size = 768
score_map_size = img_size // 4
data_root = 'data/mvtec_ad'

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
    batch_size=8,
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

test_dataloader = dict(
    batch_size=16,
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

ast_metrics = [
    'image_auroc',
    'image_auroc_mean',
    'image_auroc_max',
    'pixel_auroc',
    'image_f1max',
    'pixel_f1max',
    'image_ap',
    'pixel_ap',
    'aupro',
]

test_evaluator = [
    dict(
        type='AnomalyDetectionMetric',
        metrics=ast_metrics,
        resize_mask=score_map_size,
    ),
]
val_evaluator = test_evaluator

train_cfg = dict(by_epoch=True, max_epochs=72, val_interval=24)

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=2e-4, eps=1e-8, weight_decay=1e-5),
)

model = dict(
    type='ASTDetector',
    backbone='tf_efficientnet_b5',
    extract_layer=35,
    n_feat=304,
    map_len=24,
    n_coupling_blocks=4,
    channels_hidden_teacher=64,
    channels_hidden_student=1024,
    n_student_blocks=4,
    clamp=1.9,
    kernel_sizes=[3, 3, 3, 5],
    teacher_weight=1.0,
    student_weight=1.0,
    img_size=img_size,
    score_map_size=score_map_size,
    pos_enc=True,
    pos_enc_dim=32,
    use_gamma=True,
    training_phase='student',
    image_score_mode='mean',
)

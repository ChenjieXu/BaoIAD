"""MuSc strict official-alignment config.

Reference freeze:
- Official repository: `xrli-U/MuSc`
- Commit: `72d58ad56c0cafa2b056bd0aa7676f9c21fccbc4`
- Runtime authority: `examples/musc_main.py` + `configs/musc.yaml`
- Official CLIP mainline: `ViT-L-14-336`, `openai`, `img_resize=518`

MuSc is a zero-shot method. There is no optimizer/scheduler-driven training in
the official path; this config keeps a minimal MMEngine-compatible train shell
while freezing the official inference behavior.
"""

_base_ = [
    '../_base_/default_runtime.py',
]

data_root = 'data/mvtec_anomaly_detection'
img_size = 518

randomness = dict(seed=42, deterministic=False)

# MuSc benchmarks must run one category at a time because mutual scoring is
# computed within the unlabeled test pool of that category.
benchmark_multi_class = False
benchmark_eval_only = True
benchmark_timeout = 14400

train_pipeline = [
    dict(type='LoadImage'),
    dict(type='LoadMask'),
    # Reference CLIP preprocessing:
    # Resize((518, 518), bicubic) -> CenterCrop(518) -> ToTensor() -> OpenAI normalize.
    dict(type='OpenCLIPPreprocessAD', size=img_size),
    dict(type='PackADInputs'),
]

test_pipeline = [
    dict(type='LoadImage'),
    dict(type='LoadMask'),
    dict(type='OpenCLIPPreprocessAD', size=img_size),
    dict(type='PackADInputs'),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MVTecADDataset',
        data_root=data_root,
        split='train',
        # Keep probe/train shell runnable without class overrides. benchmark.py
        # and smoke commands override this to per-category `multi_class=False`.
        multi_class=True,
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=4,
    num_workers=0,
    persistent_workers=False,
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
    batch_size=4,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MVTecADDataset',
        data_root=data_root,
        split='test',
        multi_class=True,
        pipeline=test_pipeline,
    ),
)

model = dict(
    type='MuScDetector',
    backbone=dict(
        type='MuScCLIPBackbone',
        model_name='ViT-L-14-336',
        pretrained='openai',
        # Official config ids; runtime extraction resolves them to +1 CLIP blocks.
        feature_layers=[5, 11, 17, 23],
        image_size=img_size,
        frozen=True,
        use_ref_open_clip=True,
        require_ref_open_clip=True,
    ),
    feature_layers=[5, 11, 17, 23],
    r_list=[1, 3, 5],
    image_size=img_size,
    topmin_min=0.0,
    topmin_max=0.3,
    k_list=[1, 2, 3],
)

val_evaluator = dict(type='AnomalyDetectionMetric')
test_evaluator = dict(type='AnomalyDetectionMetric')

# Zero-shot compatibility shell only. The official MuSc path does not optimize
# model weights; benchmark.py therefore uses `benchmark_eval_only=True`.
train_cfg = dict(by_epoch=True, max_epochs=1, val_begin=1, val_interval=1)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=0.0, betas=(0.9, 0.999), weight_decay=0.0),
)

param_scheduler = []

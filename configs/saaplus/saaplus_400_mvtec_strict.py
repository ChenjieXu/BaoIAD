_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/mvtec_ad.py',
]

benchmark_eval_only = True
benchmark_multi_class = False

test_pipeline = [
    dict(type='LoadImage', to_rgb=False, keep_bgr_copy=True),
    dict(type='LoadMask'),
    dict(type='PackADInputs'),
]

model = dict(
    type='SAADetector',
    mode='saa+',
    class_name='object',
    image_size=400,
    grounding_dino_cfg=dict(
        config_path='groundingdino/config/GroundingDINO_SwinT_OGC.py',
        checkpoint='pretrained/groundingdino_swint_ogc.pth',
    ),
    sam_cfg=dict(
        model_type='vit_h',
        checkpoint='pretrained/sam_vit_h_4b8939.pth',
    ),
    saliency_backbone=dict(
        type='SAASaliencyBackbone',
        model_name='wide_resnet50_2',
        out_indices=(1, 2, 3),
        pretrained=False,
        checkpoint_path='pretrained/wide_resnet50_racm-8234f177.pth',
        frozen=True,
        image_size=1024,
    ),
    box_threshold=0.1,
    text_threshold=0.1,
    k_mask=5,
    # Keep the detector default exact, and only relax the max-area gate for
    # known borderline `pill` proposals that drift slightly vs the official
    # GDINO stack.
    box_area_tolerance=0.0,
    box_area_tolerance_overrides=dict(pill=0.003),
    # Official strict SAA+ uses confidence-prompt anomaly maps followed by
    # dataset-level normalization and image-score=max(map).
    image_score_aggregation='map_max',
    # Official `SAA/model.py` forwards the cv2-loaded BGR image directly to
    # `SamPredictor.set_image()` instead of pre-converting to RGB.
    sam_preconvert_rgb=False,
    defect_area_threshold=0.5,
)

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=1e-3, weight_decay=1e-5),
)

param_scheduler = []
train_cfg = dict(by_epoch=True, max_epochs=1, val_interval=1)

train_dataloader = dict(batch_size=1)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MVTecADDataset',
        data_root='data/mvtec_ad',
        split='test',
        multi_class=True,
        pipeline=test_pipeline,
    ),
)
test_dataloader = val_dataloader

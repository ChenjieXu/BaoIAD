_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/mvtec_ad.py',
    '../_base_/schedules/schedule_100e.py',
]

model = dict(
    type='PatchCore',
    backbone=dict(
        type='TIMMBackbone',
        model_name='wide_resnet50_2',
        pretrained=True,
        features_only=True,
        out_indices=(2, 3),  # layer2+layer3 (timm 0-indexed: 0=stem,1=layer1,2=layer2,3=layer3)
        frozen=True,
    ),
    neck=dict(
        type='MultiScalePooling',
        output_size=28,
    ),
    head=dict(
        type='MemoryBankHead',
        coreset_ratio=0.1,
        num_neighbors=9,  # Align with anomalib reference
        distance='euclidean',
        input_size=(256, 256),
        blur_sigma=4.0,
        reweight_scores=False,
        image_score_source='postprocessed',
        patch_score_neighbors=1,
        patch_score_reduction='first',
        coreset_sampling_method='approx_greedy',
        coreset_projection_dim=128,
        coreset_starting_points=10,
        coreset_device='auto',
    ),
    freeze_backbone=True,
)

os = __import__('os')

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/mvtec_ad.py',
]

clip_checkpoint = 'pretrained/open_clip/vit_b_16_plus_240-laion400m_e31-8fb26589.pt'
clip_cache_dir = 'pretrained/open_clip/.cache'
clip_pretrained = clip_checkpoint if os.path.isfile(clip_checkpoint) else 'laion400m_e31'

model = dict(
    type='WinClipDetector',
    class_name='object',
    scales=(2, 3),
    k_shot=0,
    apply_transform=False,
    backbone=dict(
        type='OpenCLIPBackbone',
        model_name='ViT-B-16-plus-240',
        pretrained=clip_pretrained,
        cache_dir=clip_cache_dir,
        frozen=True,
    ),
)

# WinCLIP is zero-/few-shot, no training needed
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=1e-3, weight_decay=1e-5),
)

param_scheduler = []

train_cfg = dict(by_epoch=True, max_epochs=1, val_interval=1)
val_cfg = dict()
test_cfg = dict(type='ADTestLoop')

_base_ = ['../_base_/default_runtime.py']

data_root = 'data/memae_video/UCSD_P2_256'
clip_length = 16
img_size = 256
dataset_name = 'UCSDped2'

train_dataloader = dict(
    batch_size=10,
    num_workers=1,
    persistent_workers=False,
    sampler=dict(
        type='MemAEOfficialOrderSampler',
        epochs=100,
        seed=1,
        in_channels=1,
        mem_dim=2000,
        shrink_thres=0.0025,
        round_up=False,
    ),
    dataset=dict(
        type='MemAEOfficialClipDataset',
        data_root=data_root,
        split='train',
        dataset_name=dataset_name,
        clip_length=clip_length,
        in_channels=1,
        img_size=img_size,
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MemAEOfficialClipDataset',
        data_root=data_root,
        split='test',
        dataset_name=dataset_name,
        clip_length=clip_length,
        in_channels=1,
        img_size=img_size,
    ),
)

val_dataloader = test_dataloader

test_evaluator = dict(type='MemAEVideoMetric')
val_evaluator = test_evaluator

model = dict(
    type='MemAEDetector',
    in_channels=1,
    frame_num=clip_length,
    clip_mode='repeat_image',
    mem_dim=2000,
    shrink_thres=0.0025,
    entropy_loss_weight=0.0002,
    temporal_reduce_mode='mean',
    image_score_mode='spatiotemporal_mean',
    loss=dict(type='MSELoss'),
    data_preprocessor=dict(type='mmengine.model.ImgDataPreprocessor'),
)

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=1e-4, weight_decay=0),
)

param_scheduler = []

randomness = dict(seed=1, deterministic=True)

train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
val_cfg = dict()
test_cfg = dict(type='ADTestLoop')

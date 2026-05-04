default_scope = 'baoiad'

custom_imports = dict(
    imports=['baoiad'],
    allow_failed_imports=False,
)

# Use ADTestLoop/ADValLoop to support deferred scoring (e.g., MuSc)
test_cfg = dict(type='ADTestLoop')
val_cfg = dict(type='ADValLoop')

custom_hooks = [dict(type='MemoryBankHook')]

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='ADVisualizationHook', enable=False),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

log_level = 'INFO'
load_from = None
resume = False

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='ADVisualizer', vis_backends=vis_backends)

# Random seed for reproducibility (matches anomalib default seed=42)
randomness = dict(seed=42, deterministic=False)

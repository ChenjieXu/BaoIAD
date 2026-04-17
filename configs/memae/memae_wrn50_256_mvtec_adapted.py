_base_ = ['./memae_wrn50_256_mvtec.py']

model = dict(
    clip_mode='centered_local_motion_window',
    clip_window_scale_min=0.98,
    clip_window_translation_max=0.02,
    image_score_mode='map_topk_mean',
    topk_ratio=0.01,
)

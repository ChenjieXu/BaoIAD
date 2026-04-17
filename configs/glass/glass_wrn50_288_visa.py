# VisA evaluation config: inherits from glass_wrn50_288_mvtec_strict.py, overrides for VisA
# Usage: python tools/benchmark.py --config configs/glass/glass_wrn50_288_visa.py \
#           --data_root data/visa --categories all --output runs/visa/glass.json

_base_ = ['./glass_wrn50_288_mvtec_strict.py']

# Benchmark mode: SC (per-category)
benchmark_multi_class = False

data_root = 'data/visa'

# VisA does not have GLASS-specific assets (fg_mask, distribution xlsx).
# Disable strict assets and foreground masking for VisA.
train_dataloader = dict(
    dataset=dict(
        type='GLASSDataset',
        data_root=data_root,
        split='train',
        cls_names=None,
        multi_class=True,
        dataset_name='visa',
        dtd_path='auto',
        fg=0,
        distribution=1,  # random distribution, no file needed
        rand_aug=1,
        strict_assets_required=False,
        fg_mask_root=None,
        distribution_meta_path=None,
        _delete_=True,
        pipeline=[dict(type='PackGLASSInputs')],
    ))

test_dataloader = dict(
    dataset=dict(
        type='VisADataset',
        data_root=data_root,
        split='test',
        multi_class=True,
        _delete_=True,
        pipeline=[dict(type='PackADInputs')],
    ))
val_dataloader = test_dataloader

# Disable strict mode in model since no distribution assets
model = dict(
    strict=False,
    distribution_meta_path=None,
)

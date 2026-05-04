# VisA evaluation config: inherits from univad_mvtec_strict.py, overrides dataset to VisA
# Usage: python tools/benchmark.py --config configs/univad/univad_mvtec_visa.py \
#           --data_root data/visa --categories all --output runs/visa/univad.json

_base_ = ['./univad_mvtec_strict.py']

# Override dataset to VisA
train_dataloader = dict(
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True))
test_dataloader = dict(
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True))
val_dataloader = dict(
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True))

# VISA has no pre-computed C3 masks — disable strict mask requirements
model = dict(
    require_mask_dir=False,
    require_heat_mask_dir=False,
    mask_dir='',
    heat_mask_dir='',
)

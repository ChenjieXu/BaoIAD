# VisA evaluation config: inherits from pni_wrn101_480_mvtec_strict.py, overrides dataset to VisA
# Usage: python tools/benchmark.py --config configs/pni/pni_wrn101_480_visa.py \
#           --data_root data/visa --categories all --output runs/visa/pni.json

_base_ = ['./pni_wrn101_480_mvtec_strict.py']

# Override dataset to VisA
data_root = 'data/visa'
train_dataloader = dict(
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True))
test_dataloader = dict(
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True))
val_dataloader = dict(
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True))

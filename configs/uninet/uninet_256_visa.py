# VisA evaluation config: inherits from uninet_256_mvtec_strict.py, overrides dataset to VisA
# Usage: python tools/benchmark.py --config configs/uninet/uninet_256_visa.py \
#           --data_root data/visa --categories all --output runs/visa/uninet.json

_base_ = ['./uninet_256_mvtec_strict.py']

# Override dataset to VisA
train_dataloader = dict(
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True))
test_dataloader = dict(
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True))
val_dataloader = dict(
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True))

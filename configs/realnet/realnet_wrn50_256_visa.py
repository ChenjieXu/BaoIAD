# VisA evaluation config: inherits from realnet_wrn50_256_mvtec_strict.py, overrides dataset to VisA
# Usage: python tools/benchmark.py --config configs/realnet/realnet_wrn50_256_visa.py \
#           --data_root data/visa --categories all --output runs/visa/realnet.json

_base_ = ['./realnet_wrn50_256_mvtec_strict.py']

# Override dataset to VisA
data_root = 'data/visa'
train_dataloader = dict(
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True))
test_dataloader = dict(
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True))
val_dataloader = dict(
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True))

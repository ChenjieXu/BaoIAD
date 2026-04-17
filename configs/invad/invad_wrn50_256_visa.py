# VisA evaluation config: inherits from invad_wrn50_256_mvtec_strict.py, overrides dataset to VisA
# Usage: python tools/benchmark.py --config configs/invad/invad_wrn50_256_visa.py \
#           --data_root data/visa --categories all --output runs/visa/invad.json

_base_ = ['./invad_wrn50_256_mvtec_strict.py']

# Benchmark mode: MC
benchmark_multi_class = True

# Override dataset to VisA
train_dataloader = dict(
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True))
test_dataloader = dict(
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True))
val_dataloader = dict(
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True))

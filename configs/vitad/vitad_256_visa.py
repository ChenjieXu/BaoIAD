# VisA evaluation config: inherits from vitad_256_mvtec_strict.py, overrides dataset to VisA
# Usage: python tools/benchmark.py --config configs/vitad/vitad_256_visa.py \
#           --data_root data/visa --categories all --output runs/visa/vitad.json

_base_ = ['./vitad_256_mvtec_strict.py']

# Benchmark mode: MC
benchmark_multi_class = True

# Disable custom train script for VisA (the exact_order script is MVTec-specific)
benchmark_train_script = None

# Override dataset to VisA
train_dataloader = dict(
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True))
test_dataloader = dict(
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True))
val_dataloader = dict(
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True))

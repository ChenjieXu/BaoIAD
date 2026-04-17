# RegAD VisA evaluation config
# Keeps RegADTrainDataset/RegADTestDataset, overrides data_root to VisA
_base_ = ['./regad_wrn50_256_mvtec_strict.py']

# Benchmark mode: SC (per-category)
benchmark_multi_class = False

# Override data roots to VisA
train_dataloader = dict(dataset=dict(data_root='data/visa'))
test_dataloader = dict(dataset=dict(data_root='data/visa'))
val_dataloader = dict(dataset=dict(data_root='data/visa'))
model = dict(data_root='data/visa')

# VisA categories don't have official support sets — allow fallback sampling
strict_require_official_support_set = False

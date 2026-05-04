"""AnomalyCLIP VisA evaluation config.

Official protocol adapted for VisA: train prompt learner on MVTec AD auxiliary
data, then evaluate on VisA dataset. (Mirrors the MVTec config which trains on
VisA and evaluates on MVTec.)
"""

_base_ = ['./anomalyclip_vitl14_336_518_mvtec_strict.py']

# Benchmark mode: MC
benchmark_multi_class = True

# Swap: eval on VisA, train on MVTec AD
data_root = 'data/visa'
aux_data_root = 'data/mvtec_ad'

# Train on MVTec AD (was aux_data_root in MVTec config)
train_dataloader = dict(
    dataset=dict(
        type='MVTecADDataset',
        data_root='data/mvtec_ad',
        split='test',
        multi_class=True,
    ),
)

# Evaluate on VisA
val_dataloader = dict(
    dataset=dict(
        type='VisADataset',
        data_root='data/visa',
        split='test',
        multi_class=True,
    ),
)
test_dataloader = val_dataloader

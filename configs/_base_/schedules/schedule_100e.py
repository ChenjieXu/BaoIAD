optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=1e-3, weight_decay=1e-5),
)

param_scheduler = [
    dict(type='CosineAnnealingLR', T_max=100, by_epoch=True, eta_min=1e-5),
]

train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=10)

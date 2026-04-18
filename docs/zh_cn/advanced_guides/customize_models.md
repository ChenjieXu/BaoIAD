# 自定义模型

## 多优化器设置

某些方法需要对不同组件使用不同的学习率：

```python
optim_wrapper = dict(
    projector=dict(
        optimizer=dict(type='Adam', lr=0.001),
    ),
    discriminator=dict(
        optimizer=dict(type='Adam', lr=0.0001),
    ),
)
```

## 自定义钩子

```python
from baoiad.registry import HOOKS
from mmengine.hooks import Hook


@HOOKS.register_module()
class MyCustomHook(Hook):

    def after_train_iter(self, runner, batch_idx, data_batch, outputs):
        # 每次训练迭代后的自定义逻辑
        pass
```

## 冻结控制

```python
model = dict(
    type='RD',
    backbone=dict(...),
    freeze_backbone=True,     # 冻结骨干网络（默认）
)
```

# BaoIAD 仓库指南

BaoIAD 是一个以源码形式分发的工业异常检测基准代码仓库。仓库本地方法清单位于 [`baoiad/method_inventory.py`](../../../baoiad/method_inventory.py)，`python tools/benchmark.py --methods all` 会选择这 37 个方法 slug。数据集、权重和可选依赖需单独获取。

方法细节见 [`configs/`](../../../configs/) 下的配置 README；实现溯源与可复现性记录见 [`docs/alignment/`](../../alignment/)。

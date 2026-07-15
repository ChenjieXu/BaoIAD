# BaoIAD 仓库指南

BaoIAD 是基于 MMEngine 的工业异常检测基准工具箱。仓库清单包含 9 个家族的 37 个方法集成，以及 10 个公开数据集的适配器。不同方法的验证深度、外部前置条件和可再分发状态不同，因此这份清单不表示所有方法都能从全新 clone 中独立复现或直接横向比较。

## 仓库范围

- [`baoiad/method_inventory.py`](../../baoiad/method_inventory.py) 是 37 方法结构清单；`python tools/benchmark.py --methods all` 会选择这些方法 slug。
- [`configs/`](../../configs/) 下的方法 README 提供配置入口和方法特有说明。
- [`docs/alignment/`](../alignment/) 记录实现来源、差异、运行前置条件和已知局限。
- [`method_status.json`](../alignment/method_status.json) 是“部分验证”和“历史证据”状态的机器可读事实源。原始研究 artifact 不随本公开发布分发。

BaoIAD 不随仓分发数据集、预训练权重或所有可选依赖。用户需按原始条款获取外部资产。

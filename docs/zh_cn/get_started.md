# BaoIAD 仓库指南

BaoIAD 是一个工业异常检测基准代码仓库。仓库本地方法清单位于 [`baoiad/method_inventory.py`](../../baoiad/method_inventory.py)，`python tools/benchmark.py --methods all` 会选择这 37 个方法 slug。

方法细节见 [`configs/`](../../configs/) 下的配置 README；实现溯源与可复现性记录见 [`docs/alignment/`](../alignment/)。

## 源码安装

```bash
git clone https://github.com/Baosight-xVue/BaoIAD.git
cd BaoIAD
pip install -e .
```

不同方法可能需要 `pyproject.toml` 中定义的可选 extra。数据集和外部预训练资产不随仓分发，需用户按各自条款获取。

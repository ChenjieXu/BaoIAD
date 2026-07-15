# BaoIAD 仓库指南

BaoIAD 是一个工业异常检测基准代码仓库。仓库本地方法清单位于 [`baoiad/method_inventory.py`](../../baoiad/method_inventory.py)，`python tools/benchmark.py --methods all` 会选择这 37 个方法 slug。

方法细节见 [`configs/`](../../configs/) 下的配置 README；实现溯源与可复现性记录见 [`docs/alignment/`](../alignment/)。

本发布在 Python 3.10 和 3.12 上验证。核心安装使用 `mmcv-lite>=2.0`、`mmengine>=0.10` 和 `torch>=2.0`；对外声明 GPU 支持前，须另行验证目标 CUDA 构建。

## 源码安装

```bash
git clone https://github.com/Baosight-xVue/BaoIAD.git
cd BaoIAD
pip install -e .
```

不同方法可能需要 `pyproject.toml` 中定义的可选 extra。数据集和外部预训练资产不随仓分发，需用户按各自条款获取。

常用可选依赖组如下：

```bash
pip install -e ".[flow]"          # FrEIA 等归一化流依赖
pip install -e ".[vl]"            # open_clip_torch
pip install -e ".[saa]"           # groundingdino、segment-anything
pip install -e ".[geomloss]"      # 最优传输损失
pip install -e ".[glass]"         # pandas、openpyxl
pip install -e ".[visualization]" # matplotlib
pip install -e ".[faiss-cpu]"     # faiss-cpu
pip install -e ".[all]"           # 上述运行时可选组
pip install -e ".[dev]"           # pytest、ruff、pre-commit
```

## 安装验证

安装后运行轻量本地检查：

```bash
python tools/check_install.py
python tools/check_method_inventory.py
```

`check_install.py` 会输出 BaoIAD、Python、PyTorch、MMEngine 与 MMCV 版本，解析数据根目录和缓存目录，并报告各可选依赖组是否可用。缺失可选组仅作提示，不会使有效的核心安装失败。

这是一道仅使用 CPU 的本地发布门禁。它不会下载模型、读取数据集内容、加载 checkpoint、启动训练、查询 CUDA/MPS 或执行 GPU 计算，因此通过自检**不代表**已经完成 GPU 验证。对外声明 GPU 支持前，必须在目标 CUDA/PyTorch/MMCV 组合上，以所选方法、资产和数据集另行执行独立 GPU smoke gate。

`python tools/check_install.py --offline` 会为自检进程启用 BaoIAD 及受支持模型仓的离线环境变量；即使不传该参数，自检本身也不会访问网络。`train.py`、`test.py` 和 `benchmark.py` 的同名参数会阻止受支持的下载路径，此时必须事先在本地准备数据集和外部资产。

`check_install.py` 不需要也不接受 `--trusted-checkpoint`，因为它不会加载 checkpoint。只有在 `train.py`、`test.py` 或 `benchmark.py` 中使用经过可信来源和完整性校验的旧式 pickle checkpoint 时，才应显式传入该参数；反序列化可能执行任意代码。应优先使用可安全加载的 checkpoint 格式以及默认的受限策略。

所有受支持的 PyTorch 版本都会以受限方式加载仅含 tensor 的 `.pth` 和 `.safetensors`。PyTorch 2.6 及以上还可通过窄化的 safe-globals 清单恢复 BaoIAD 已知的 MMEngine `HistoryBuffer` 元数据。在 PyTorch 2.0–2.5 上，默认会拒绝含该元数据的标准 MMEngine resume/evaluation checkpoint；独立核验来源和完整性后，应优先迁移到 PyTorch 2.6 及以上，否则才可对该文件显式使用 `--trusted-checkpoint`。

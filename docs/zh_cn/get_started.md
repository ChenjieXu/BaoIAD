# 快速开始

## 安装

### 前置条件

- Python >= 3.9
- PyTorch >= 2.0（建议支持 CUDA）
- [MMEngine](https://github.com/open-mmlab/mmengine) >= 0.10
- [MMCV](https://github.com/open-mmlab/mmcv) >= 2.0

### 安装 BaoIAD

```bash
git clone https://github.com/xxx/BaoIAD.git
cd BaoIAD
pip install -e .
```

### 可选依赖

BaoIAD 针对特定方法族使用可选依赖：

```bash
pip install -e ".[flow]"        # 归一化流方法（CSFlow、FastFlow、CFlow 等）
pip install -e ".[vl]"          # 视觉语言方法（WinCLIP、AnomalyCLIP 等）
pip install -e ".[saa]"         # SAA/SAA+（需要 groundingdino + segment_anything）
pip install -e ".[geomloss]"    # MuSc、RD++（最优传输损失）
pip install -e ".[imgaug]"      # DRAEM、DeSTSeg（异常合成）
pip install -e ".[mmpretrain]"  # UniNet、UniVAD
pip install -e ".[faiss-cpu]"   # PatchCore、PaDiM（快速最近邻搜索）
pip install -e ".[all]"         # 安装所有可选依赖
```

开发环境：

```bash
pip install -e ".[dev]"         # pytest、ruff、pre-commit
```

### 环境配置

设置数据缓存和模型下载的环境变量：

```bash
source tools/env.sh
```

或手动设置：

```bash
export BAOIAD_DATA_ROOT=/path/to/data   # 默认：./data
export HF_HOME=/path/to/hf_cache        # HuggingFace 缓存
export TORCH_HOME=/path/to/torch_cache  # PyTorch 模型缓存
```

## 验证安装

```python
import baoiad
print(baoiad.__version__)  # 0.1.0
```

快速功能测试：

```bash
pytest tests/ -k "test_patchcore" -x
```

## 快速上手

### 训练

在 MVTec AD（全部 15 个类别）上训练 PatchCore：

```bash
python tools/train.py configs/patchcore/patchcore_wrn50_256_mvtec_strict.py \
    --work-dir runs/patchcore
```

训练单个类别：

```bash
python tools/train.py configs/patchcore/patchcore_wrn50_256_mvtec_strict.py \
    --work-dir runs/patchcore_bottle \
    --cfg-options train_dataloader.dataset.cls_names="['bottle']" train_dataloader.dataset.multi_class=False
```

### 测试

使用训练好的检查点进行测试：

```bash
python tools/test.py configs/patchcore/patchcore_wrn50_256_mvtec_strict.py \
    runs/patchcore/best.pth
```

### 基准测试

跨方法和类别运行基准测试：

```bash
python tools/benchmark.py \
    --data_root data/mvtec_ad \
    --methods patchcore rd \
    --categories all \
    --output runs/benchmark_results.json
```

## 常见问题

### MMEngine/MMCV 安装

MMCV 需要从源码编译自定义 CUDA 算子。如果遇到构建错误：

```bash
pip install mmcv -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
```

请根据您的环境调整 CUDA/PyTorch 版本 URL。

### FrEIA 未找到

归一化流方法（CSFlow、FastFlow、CFlow、UFlow、DifferNet、PyramidFlow、AST）需要 FrEIA：

```bash
pip install -e ".[flow]"
```

### CUDA 内存不足

对于大模型或高分辨率图像：

- 减小批量大小：`--cfg-options train_dataloader.batch_size=8`
- 使用梯度累积：`--cfg-options optim_wrapper.accumulative_counts=4`
- 强制使用 CPU：`--cpu`（不推荐用于训练）

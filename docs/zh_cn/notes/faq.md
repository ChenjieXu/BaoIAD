# 常见问题

## 安装

**Q: 如何安装带 CUDA 支持的 MMCV？**

A: 从与您的 CUDA 和 PyTorch 版本匹配的预构建 wheel 索引安装：

```bash
pip install mmcv -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
```

**Q: 使用流方法时 FrEIA 导入失败。**

A: 安装流可选依赖：`pip install -e ".[flow]"`

**Q: 使用视觉语言方法时 open_clip 导入失败。**

A: 安装 VL 可选依赖：`pip install -e ".[vl]"`

## 训练

**Q: 记忆库方法（PatchCore、SPADE）训练非常快，这正常吗？**

A: 是的。记忆库方法在"训练"期间只进行特征提取。实际的记忆库由 `MemoryBankHook` 在训练循环结束后构建。

**Q: 如何训练特定类别？**

A: 覆盖类别列表并设置 `multi_class=False`：

```bash
python tools/train.py <config> --work-dir runs/bottle \
    --cfg-options train_dataloader.dataset.cls_names="['bottle']" train_dataloader.dataset.multi_class=False
```

**Q: DRAEM/DeSTSeg 训练因 DTD 路径错误而失败。**

A: 这些方法需要 DTD（可描述纹理数据集）进行异常合成。在模型配置中设置 `dtd_path='auto'` 或提供显式路径。

## 对齐

**Q: PaDiM 结果在不同运行之间变化。为什么？**

A: PaDiM 使用随机投影进行降维，使其对随机种子敏感。确保始终设置 `randomness.seed=42`。

**Q: 哪些方法支持多类别训练？**

A: UniAD、ViTAD、InvAD 和 MambaAD 支持 `multi_class=True` 在所有类别上训练单一模型。大多数其他方法每个类别训练一个模型。

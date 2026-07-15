# 发布流程

本文定义 BaoIAD 组织仓的发布规则方案，不表示 GitHub 分支保护、
私密安全报告、ReadTheDocs、Zenodo 或 Release 权限已经完成生产配置。
这些外部变更必须由指定 owner 执行并保留审批证据。

## 角色与审查责任

| 角色 | 必须承担的责任 |
|---|---|
| 技术维护者 | 代码/配置正确性、测试范围、兼容性与方法声明 |
| Release owner | exact commit 门禁、发布说明、tag/Release 执行与回滚协调 |
| 法务/OSS owner | 许可证、溯源、再分发和第三方处置 |
| 品牌 owner | 公开身份、顶层 README、媒体与 WAIC 文案 |
| 安全 owner | 私密报告渠道、安全分级和安全修复审批 |
| 发布值守 owner 及 backup | 发布期间监控安装、文档和可复现 bug |

所有 PR 至少由一名非作者 reviewer 批准；发布敏感文件还需相应角色
审查。在组织提供真实 GitHub 用户/团队并验证分支保护可解析之前，
不得创建虚构的 `CODEOWNERS`。

## PR 必需检查

`master` 的 P0 required-check 合同方案为：

1. `lint`
2. `release-policy`
3. `core-offline (3.10)`
4. `core-offline (3.12)`
5. `docs-en`
6. `docs-zh`

workflow 的稳定 job 名必须与分支规则中的 context 完全一致。core job 使用
已提交的 CPU constraints，并包含 core import、方法清单、公开发布检查和
无数据依赖 CPU 测试；双语 Sphinx warning 直接导致失败。

optional-extra/registry、clean-install、定时外部链接、network、slow 和 GPU job
必须分开展示。只有稳定的 offline CPU job 可作为 PR 必需检查。
开启分支保护前，Release owner 必须从组织仓成功运行中记录精确 context。

## 分支保护方案

组织管理员应为 `Baosight-xVue/BaoIAD:master` 开启：

- 必须通过 PR 修改；
- 至少一名 reviewer 批准；
- P0 checks 全部通过且与目标分支同步；
- 相关修改后驳回过期批准；
- 必须解决 review conversation；
- 禁止 force push 和删除分支；
- 管理员是否可绕过由组织政策明确决定并审计。

仓库文件不会自动更改上述生产设置。配置后还必须用验证 PR 证明直接
push 和失败 check 会被阻止。

## GPU 证据是独立门禁

CPU 发布检查不能验证 CUDA。真实 GPU 证据必须记录：

- GPU 型号、驱动、CUDA runtime、Python、PyTorch、TorchVision、
  精确的 `mmcv` 或 `mmcv-lite` 包名/版本和 commit；
- 需要的 compiled CUDA ops 是否存在，或为何不适用；
- 关键方法训练/推理 smoke 命令与结果；
- 峰值 allocated/reserved 显存和 OOM 情况；
- 仅含占位符的命令和无凭据、原始数据集/工作目录/checkpoint 字段、
  私有绝对路径或 file URI 的日志引用。

没有真实 CUDA 设备日志时，状态必须为 **GPU 未验证**，不得标记为 green，
也不得声称已验证 GPU 训练、CUDA ops、峰值显存或 37 方法端到端。
只有公开发布范围明确排除 GPU 验证，且 go/no-go owner 接受该限制时，
Release owner 才可在其他门禁全部通过后继续。

[真实 GPU 发布验证](../../release/gpu_validation.md)说明了手动 workflow、
证据合同和 G007 必需门禁。

## Exact-commit pre-tag 门禁

tag 候选的 exact merged commit 必须完成：

1. required CI 全绿，且此后代码未变更。
2. Python 3.10/3.12 使用同一份 CPU constraints clean install。
3. exact diff、allowlist、禁止跟踪路径、文件大小、本地链接、secret scan
   和研究工作区 before/after 证据通过。
4. 方法清单、溯源、许可证、资产授权和公开声明无 release-blocking 未决项。
5. 中英文 README/ReadTheDocs 渲染获得技术和品牌审批。
6. 法务/OSS、品牌、技术、安全、Release owner 审批已记录，tag/Release、
   ReadTheDocs 和 Zenodo 权限 preflight 通过。
7. release notes、兼容边界、支持范围、值守与热修复流程已准备。

[外部审批表](../../release/external_approvals.json)中的 pending release blocker 必须保持发布阻断。
特别是 `APP-SECURITY-CHANNEL` 未通过时，公开发布的安全门禁不得通过。
任一 yellow/red 状态都不得打 tag；修复后在新 exact commit 上重跑全部门禁。

## 发布、热修复与回滚

go 结论记录后，从审批 commit 创建 annotated tag，只发布审计通过的源码和
说明；数据集、未审计权重、凭据和实验压缩包不得附加。ReadTheDocs、
Zenodo 和 GitHub Release 只能由已批准 owner 操作。

未发布候选使用正常 revert/fix PR 并重跑门禁。已发布回归从不可变 tag
派生最小修复分支，通过同样 P0 gate 后发布新 patch tag（例如 `v1.1.1`）。
禁止 force push、删除/移动已发布 tag 或重定向旧 Release。

安全修复遵循 [SECURITY.md](../../../SECURITY.md)，贡献者检查见
[CONTRIBUTING.md](../../../CONTRIBUTING.md)。

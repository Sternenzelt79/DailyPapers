---
type: concept
aliases: [零初始化门控, Zero-initialized Gating, Zero-init Gate]
---

# Zero-initialized Gating

## 定义
将可学习门控参数（标量或向量）初始化为零，用于控制新增分支（如条件输入、额外模态）对主干网络的影响程度。初始状态下新分支贡献为零，随训练逐渐增大。

## 数学形式
$$e_{\text{fused}} = e_{\text{main}} + g \cdot e_{\text{branch}}, \quad g \leftarrow 0$$

其中 $g$ 为可学习标量，初始化为 $0$，训练过程中梯度更新。

## 核心要点
1. **训练稳定性**: 初始化为零确保训练初期行为与主干预训练模型一致，避免随机初始化带来的训练不稳定
2. **ControlNet 范式**: 广泛用于 ControlNet 等条件生成模型，使条件分支平滑接入预训练模型
3. **潜在问题**: MaskWAM 的消融实验表明，当用于 Mask 条件注入时，门控易学到始终忽略 Mask 信号（$g \approx 0$），导致条件失效

## 代表工作
- [[MaskWAM]]: 作为弃用的备选设计被讨论，实验证明对 Mask 条件注入效果不佳，改用直接通道拼接+监督

## 相关概念
- [[Diffusion Transformer]]
- [[Instance Segmentation Mask]]
- [[Mixture of Transformers]]

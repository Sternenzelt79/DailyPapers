---
type: concept
aliases: [Test-Time Training, 测试时训练]
---

# TTT

## 定义
Test-Time Training，在测试/推理阶段对模型参数进行快速更新的技术，通过自监督辅助任务让模型适应当前输入分布。

## 数学形式
$$\theta' = \theta - \alpha \nabla_\theta \mathcal{L}_{\text{self-sup}}(x_{\text{test}})$$

## 核心要点
1. 不需要测试标签，使用自监督损失（如旋转预测、掩码重建）
2. 分为 online TTT（逐样本更新）和 batch TTT（批量更新）
3. 在 [[MemoryWAM]] 等序列模型中用于高效历史压缩

## 代表工作
- Sun et al. (2020): 原始 TTT 论文
- [[MemoryWAM]]: 应用 TTT 思路做 gist token 压缩

## 相关概念
- [[MemoryWAM]]

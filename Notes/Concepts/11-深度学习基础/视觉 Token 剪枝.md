---
type: concept
aliases: [Visual Token Pruning, Token Pruning, 视觉token剪枝]
---

# 视觉 Token 剪枝

## 定义
视觉 Token 剪枝（Visual Token Pruning）是一种在 Vision-Language Model（VLM）或 Vision-Language-Action（VLA）推理阶段动态减少视觉 Token 数量的方法，通过丢弃重要性低的视觉 Token 来降低 Transformer 自注意力的二次计算开销。

## 数学形式

给定视觉 Token 集合 $\mathbf{E}_v \in \mathbb{R}^{M \times d}$，剪枝目标为：

$$
\min_f \mathcal{L}(\mathcal{P}, \tilde{\mathcal{P}}) \quad \text{s.t.} \quad |f(\mathbf{E}_v)| = \tilde{M}
$$

其中 $f$ 为 Token 选择函数，$\tilde{M} < M$ 为目标保留数量，$\mathcal{P}$ 和 $\tilde{\mathcal{P}}$ 分别为原始和剪枝后的模型输出分布。

## 核心要点
1. **计算瓶颈**：视觉 Token 数量（~256×n）远超文本 Token（~30-50）和动作 Token（~7-56），是 Transformer 计算的主要开销来源
2. **重要性评估**：通常基于 Attention Score 排名（如 Prefill 注意力均值）确定 Token 的保留优先级
3. **VLM vs VLA 差异**：VLM 剪枝关注语义显著性，VLA 还需兼顾动作相关性（存在语义-动作鸿沟）
4. **训练无关**：多数方法为即插即用模块，无需额外微调

## 代表工作
- [[FastV]]: 基于 Prefill 注意力的 VLM 视觉 Token 剪枝先驱方法
- [[VLA-Pruner]]: 首个针对 VLA 推理的双重重要性（语义+动作）Token 剪枝方法
- [[DivPrune]]: 基于多样性的 Token 剪枝方法
- [[SparseVLM]]: 基于 Text-to-Vision 注意力的剪枝方法

## 相关概念
- [[多头自注意力]]
- [[交叉注意力]]
- [[最大最小多样性问题]]
- [[指数移动平均]]

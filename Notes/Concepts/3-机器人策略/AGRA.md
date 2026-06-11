---
type: concept
aliases: [Action-Grounded Representation Alignment, 动作基础表示对齐]
---

# AGRA

## 定义

AGRA（Action-Grounded Representation Alignment）是一种训练目标函数，通过将视频扩散模型的中间层特征与冻结基础视觉编码器（如 DINOv2）的空间语义特征对齐，使 [[World-Action Model]] 的世界-动作接口更具"可操作性"。

## 数学形式

$$
\mathcal{L}_{AGRA} = -\frac{1}{KT_vH_vW_v} \sum_{k=1}^{K}\sum_{t=1}^{T_v}\sum_{u=1}^{H_v}\sum_{v=1}^{W_v} \cos(\mathbf{Z}_{k,t,u,v}, \tilde{Y}_{t,u,v})
$$

其中 $\mathbf{Z}_k = P_k(\mathbf{H}^{vid}_{l_k})$ 是投影后的视频扩散中间特征，$\tilde{Y}$ 是插值后的 [[DINOv2]] 目标特征。

## 核心要点

1. **问题**: [[World-Action Model]] 中视觉预测准确 ≠ 动作提取准确，动作解码器无法聚焦任务相关交互区域（action-grounding gap）
2. **技术灵感**: 将 [[REPA]] 对齐技术从"改善生成质量"重新定向为"接口正则化器"
3. **实现**: 均匀选取 Video DiT 多层（最优约 1/3 网络深度），经轻量投影层对齐 [[DINOv2]] 特征
4. **效果**: 真实人形机器人操作成功率从 34% 提升至 80%，OOD 泛化提升 27-32%

## 代表工作

- [[AGRA (Paper)]]: Making Foresight Actionable: Repurposing Representation Alignment in World Action Models (2026)

## 相关概念

- [[World-Action Model]]
- [[REPA]]
- [[DINOv2]]
- [[Flow Matching]]
- [[Cross-Attention]]

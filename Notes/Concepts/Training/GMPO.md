---
type: concept
aliases: [GMPO, Group Muon Policy Optimization]
---

# GMPO

## 定义
Group Muon Policy Optimization，对 Muon 优化器的改进版本，专为 VLA fine-tuning 和 RLVR（Reinforcement Learning for Visual Reasoning）设计，通过高通滤波修复 Muon 在 cross-modality 微调和 RLVR 中的谱失效问题。

## 数学形式
$$\mathbf{M}_t^{\text{HP}} = \mathbf{M}_t - \mathbf{U}_k \Sigma_k \mathbf{V}_k^\top$$

从动量矩阵 $\mathbf{M}_t$ 中移除低 SNR 的尾部奇异值分量（低通成分），保留高频信息量大的部分，再进行 Newton-Schulz 正交化。

## 核心要点
1. **诊断**：Muon 的均匀谱白化在 VLA 微调时破坏 cross-modality 梯度方向；在 RLVR 中低 SNR 尾部放大噪声
2. **修复**：高通滤波 + 分组 NS 迭代，让 Muon 在非预训练场景下稳定工作
3. 可替代 GRPO + AdamW 组合中的优化器部分
4. 在 LIBERO 和 DROID 数据集上验证 VLA 微调效果

## 代表工作
- [[GMPO]] (Rethinking Muon Beyond Pretraining): 提出并验证 GMPO，分析两类谱失效场景

## 相关概念
- [[GRPO]]
- [[OpenVLA-OFT]]
- [[Agentic-VLA]]

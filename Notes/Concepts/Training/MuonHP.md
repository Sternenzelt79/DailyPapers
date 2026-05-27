---
type: concept
aliases: [High-Pass Muon, Muon High-Pass]
---

# MuonHP

## 定义
MuonHP（High-Pass Muon）是对 Muon 优化器的改进，仅对梯度动量矩阵的高频/主奇异值子空间做 Newton-Schulz 谱正交化，低频噪声通道保留 [[AdamW]] 更新，解决 Muon 在 VLA 微调和 RLVR 场景下的谱失真问题。

## 数学形式
$$
\theta_{t+1} = \theta_t - \eta \cdot \text{NS}(\mathbf{M}^{\text{high}}) - \eta \cdot \text{AdamW}(\mathbf{M}^{\text{low}})
$$
其中 $\mathbf{M}^{\text{high}}$ 为梯度动量矩阵的高奇异值子空间，$\text{NS}(\cdot)$ 为 Newton-Schulz 正交化。

## 核心要点
1. 原始 Muon 将所有奇异值归一化，在微调时过度约束任务相关方向（谱失真）
2. MuonHP 用阈值分离"信息头"和"噪声尾"，只对高频部分做正交化
3. 实验在 VLA（LIBERO）和 RLVR（GRPO）两个场景验证

## 代表工作
- Rethinking Muon Beyond Pretraining (arXiv 2605.19282): MuonHP 的提出论文

## 相关概念
- [[AdamW]]
- [[GRPO]]
- [[Reinforcement Learning]]

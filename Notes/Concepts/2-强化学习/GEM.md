---
type: concept
aliases: [GEM, Gradient Episodic Memory, 梯度情节记忆]
---

# GEM (Gradient Episodic Memory)

## 定义
GEM 是一种持续学习（Continual Learning）方法，通过维护旧任务的少量样本记忆缓冲区，并在学习新任务时对梯度施加约束（确保在旧任务样本上的 loss 不增加），从而缓解灾难性遗忘。

## 数学形式
$$\min_\theta \mathcal{L}(\theta; \mathcal{D}_\text{new}) \quad \text{s.t.} \quad \mathcal{L}(\theta; \mathcal{M}_k) \leq \mathcal{L}(\theta_\text{old}; \mathcal{M}_k), \; \forall k < t$$

其中 $\mathcal{M}_k$ 是第 $k$ 个旧任务的记忆缓冲区。

## 核心要点
1. 相比简单 Experience Replay（ER），GEM 的约束更强：不仅重放旧样本，还在梯度层面保证不遗忘
2. 通过二次规划（QP）将梯度投影到满足约束的可行域
3. 计算开销比 ER 高（需要 QP 求解），但遗忘抑制效果通常更好
4. 在 VLA 持续学习研究中与 ER、FT 并列作为 baseline

## 在 VLA 中的应用
VLA-CL 研究发现 GEM 在真实机器人数据流上的遗忘抑制效果，对比 ER 的优劣取决于任务间的梯度冲突程度。

## 代表工作
- Lopez-Paz & Ranzato (2017): GEM 原始论文
- [[VLA-CL]]: 在 VLA 连续学习场景中评测 GEM

## 相关概念
- [[Reinforcement Learning]]
- [[SFT]]
- [[Cross-Task Transfer]]

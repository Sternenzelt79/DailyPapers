---
type: concept
aliases: [AdamW优化器, Adam with Weight Decay]
---

# AdamW

## 定义
Adam 优化器的改进版，将权重衰减（L2正则化）与梯度更新解耦，修正了原版 Adam 中 L2 正则化的错误实现。

## 数学形式
$$\theta_{t+1} = \theta_t - \eta \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t\right)$$

其中 $\hat{m}_t$、$\hat{v}_t$ 为一阶/二阶矩估计，$\lambda$ 为权重衰减系数。

## 核心要点
1. 将 L2 正则化直接应用于参数更新（weight decay），而非梯度
2. 避免了 Adam 中 adaptive learning rate 对 L2 正则化效果的干扰
3. 是 Transformer 预训练和大模型微调的默认优化器
4. 与 [[GRPO]]、[[RLHF]] 等 RL 训练配合使用时，稳定性优于更激进的优化器（如 [[Muon]]）

## 代表工作
- Loshchilov & Hutter (2019): Decoupled Weight Decay Regularization

## 相关概念
- [[GRPO]]
- [[SFT]]
- [[LoRA]]

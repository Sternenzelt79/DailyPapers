---
type: concept
aliases: [CRC, Conformal Risk Control, 保形风险控制]
---

# CRC（Conformal Risk Control）

## 定义
基于保形预测（Conformal Prediction）的统计框架，为机器学习模型提供有分布自由（distribution-free）保证的风险控制，通过校准阈值使模型在测试集上的期望风险不超过预设上界。

## 数学形式
$$\hat{q} = \inf\left\{q : \frac{1}{n+1}\left(1 + \sum_{i=1}^n \mathbf{1}[s_i \leq q]\right) \geq 1-\alpha\right\}$$

其中 $s_i$ 为校准集上的 score，$\alpha$ 为目标风险水平，$\hat{q}$ 为校准后阈值。

## 核心要点
1. **分布自由**：不假设数据分布，只要求 exchangeability（可交换性）
2. 通过校准集（cal set）计算阈值，保证测试集上的风险有界
3. Per-task 变体对每个任务单独校准阈值，比全局阈值更精确
4. 用于 VLA 弃权决策：当所有 K 个候选动作风险都高于阈值时触发弃权

## 代表工作
- [[BOKBO]]: 将 CRC 用于 VLA 策略的校准弃权机制，提供分布自由的安全保证

## 相关概念
- [[Adversarial Perturbation]]
- [[GRPO]]

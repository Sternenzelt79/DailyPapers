---
type: concept
aliases: [KL Token-Prediction Loss, KL Auxiliary Loss, L_KL]
---

# KL 辅助损失

## 定义

KL 辅助损失（$\mathcal{L}_{\text{KL}}$）是 NextLat 的辅助训练目标：用冻结输出头分别对真实隐状态和预测隐状态生成 token 分布，通过 KL 散度约束两者在输出分布层面的等价性，确保[[转移一致性]]不仅在几何空间成立，也在语义（token 概率）层面成立。

## 数学形式

$$
\mathcal{L}_{\text{KL}}(\theta, \psi; d) = \mathbb{E}_t \left[ \frac{1}{d} \sum_{i=1}^{d} D_{\text{KL}}\!\Bigl(p_\theta^{sg}(\cdot \mid \text{sg}[h_{t+i}]) \;\|\; p_\theta^{sg}(\cdot \mid \hat{h}_{t+i})\Bigr) \right]
$$

## 核心要点

1. **语义层面对齐**：单纯的 SmoothL1 回归仅保证几何近似；KL 损失进一步保证预测隐状态与真实隐状态在 token 输出分布上等价
2. **知识蒸馏风格**：输出头参数固定（stop-gradient），类似[[知识蒸馏]]中的软标签监督
3. **防止几何扭曲**：避免模型学到一种"几何上近似但语义上无意义"的潜在表示
4. **联合 $\mathcal{L}_{\text{next-h}}$**：两者共同近似[[转移一致性]]约束

## 代表工作

- [[NextLat]]: 联合 $\mathcal{L}_{\text{next-h}}$ 构成完整 NextLat 辅助目标

## 相关概念

- [[KL Divergence]]
- [[知识蒸馏]]
- [[停止梯度（Stop-Gradient）]]
- [[下一隐状态损失]]

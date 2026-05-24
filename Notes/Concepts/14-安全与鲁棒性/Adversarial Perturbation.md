---
type: concept
aliases: [对抗扰动, 对抗样本, Adversarial Examples]
---

# Adversarial Perturbation

## 定义

在输入数据上添加人眼难以察觉但能使模型产生错误输出的微小扰动；在鲁棒性训练中，也用于主动生成最坏情况输入以增强模型的抗干扰能力。

## 数学形式

基于梯度的有界扰动（FGSM 风格）：

$$
\tilde{x} = x + \epsilon \cdot \text{sign}(\nabla_x \mathcal{L})
$$

相对预算版本（RoVLA 采用）：

$$
\tilde{x} = x + \eta \frac{\nabla_x \mathcal{L}}{\|\nabla_x \mathcal{L}\|_2}, \quad \eta = \min(\alpha, \varepsilon_{\text{adv}} \|x\|)
$$

## 核心要点

1. **攻击视角**：寻找使模型输出错误的最小扰动，评估模型脆弱性
2. **防御视角（对抗训练）**：将对抗样本加入训练集，提升模型对扰动的鲁棒性
3. **相对预算**比固定步长更自然，对不同尺度的特征都有效
4. 在 VLA 领域，对视觉特征和本体状态扰动可提升策略的观测鲁棒性

## 代表工作

- [[RoVLA]]: 对视觉特征和本体状态施加相对预算对抗扰动，实现 [[Observational Consistency]]

## 相关概念

- [[Observational Consistency]]
- [[Evolutionary Consistency]]
- [[Vision-Language-Action Model]]

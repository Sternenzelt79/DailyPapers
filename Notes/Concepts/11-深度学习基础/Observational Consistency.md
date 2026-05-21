---
type: concept
aliases: [观测一致性, OC]
---

# Observational Consistency

## 定义

对模型的输入（视觉特征、状态）施加对抗扰动，强制扰动后的预测与干净输入的预测保持一致，从而提升模型对观测噪声和视觉变化的鲁棒性。

## 数学形式

$$
\mathcal{L}_{\text{OC}} = \frac{1}{2} \sum_{i=1}^{2} \left\|\hat{\mathbf{v}}_{\text{pert}}^{\tau_i} - \text{sg}\left(\hat{\mathbf{v}}_{\text{clean}}^{\tau_i}\right)\right\|_2^2
$$

对抗扰动生成：

$$
\tilde{v}_t = v_t + \eta_v \frac{\nabla_{v_t} \mathcal{L}_{\text{EC}}}{\|\nabla_{v_t} \mathcal{L}_{\text{EC}}\|_2}, \quad \eta_v = \min(\alpha, \varepsilon_{\text{adv}} \|v_t\|_2)
$$

## 核心要点

1. 扰动方向沿 [[Evolutionary Consistency]] 损失的梯度，使扰动针对模型最敏感的方向
2. 使用 stop-gradient (`sg`) 防止梯度回传影响干净预测路径
3. 扰动预算相对于特征范数自适应（$\varepsilon_{\text{adv}} \|v_t\|_2$），比固定步长更稳定

## 代表工作

- [[RoVLA]]: 将观测一致性用于 VLA 鲁棒性训练，改善视觉扰动和状态噪声场景

## 相关概念

- [[Evolutionary Consistency]]
- [[Adversarial Perturbation]]
- [[Vision-Language-Action Model]]

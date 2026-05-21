---
type: concept
aliases: [演化一致性, EC]
---

# Evolutionary Consistency

## 定义

在扩散模型或流匹配模型的去噪过程中，强制模型在两个不同演化时间步下对同一输入产生一致的预测，抑制去噪轨迹中的表示波动。

## 数学形式

$$
\mathcal{L}_{\text{EC}} = \left\|\hat{\mathbf{v}}_{\text{clean}}^{\tau_1} - \hat{\mathbf{v}}_{\text{clean}}^{\tau_2}\right\|_2^2
$$

其中 $\tau_1, \tau_2$ 为在非均匀 Beta 分布下采样的两个演化时间步。

## 核心要点

1. 通过对比同一输入在两个不同时间步下的预测，消除去噪过程中的噪声波动
2. 非均匀时间步采样（Beta 分布）偏向较小时间步，因为噪声阶段的预测方差更大
3. 是 [[Observational Consistency]] 的前置步骤，OC 使用 EC 的梯度方向生成对抗扰动

## 代表工作

- [[RoVLA]]: 首次将演化一致性引入 VLA 策略鲁棒性训练

## 相关概念

- [[Observational Consistency]]
- [[Conditional Flow Matching]]
- [[Flow Matching]]
- [[Diffusion Transformer]]

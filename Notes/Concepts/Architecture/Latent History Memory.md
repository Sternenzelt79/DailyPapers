---
type: concept
aliases: [潜在历史记忆, Latent History, 历史记忆]
---

# Latent History Memory

## 定义

Latent History Memory 是一种在生成式世界模型中维护跨推理轮次时序上下文的机制，通过将最近若干帧历史观测编码为潜在空间向量，在长序列滚动生成中保持时序一致性。

## 数学形式

$$
Z_t^{v,h} = E_\phi\left(\{x_t^v\}_{t \in \mathcal{I}_t^h}\right), \quad \mathcal{H}_t = \{Z_t^{v,h}\}_{v \in \mathcal{V}}
$$

其中 $E_\phi$ 为 VAE 编码器，$\mathcal{I}_t^h$ 为历史帧索引集合，$\mathcal{V}$ 为相机视角集合。

## 核心要点

1. **跨轮次上下文传递**: 在 policy-in-the-loop 评估中，每轮世界模型预测结束后，将最近 $H_h$ 帧编码进历史记忆，供下一轮生成时使用
2. **多视角一致性**: 对所有相机视角分别编码历史，确保跨视角时序一致
3. **关键性验证**: 消融实验表明，去除历史记忆后 LPIPS 恶化超过 200%

## 代表工作

- [[PiL-World]]: 提出该机制用于 VLA 闭合循环评估，历史帧数 $H_h=5$

## 相关概念

- [[VAE]]
- [[世界模型]]
- [[扩散模型]]

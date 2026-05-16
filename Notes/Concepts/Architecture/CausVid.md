---
type: concept
aliases: [CausVid, Causal Video Diffusion]
---

# CausVid

## 定义
CausVid 是一类因果自回归视频扩散模型，通过将双向 teacher 模型蒸馏为因果（单向注意力）few-step 模型，实现流式实时视频生成。

## 数学形式
$$p_\theta(x_{1:T}) = \prod_{t=1}^{T} p_\theta(x_t \mid x_{<t})$$

因果分解使得模型可以逐帧外推，不需要未来帧信息。

## 核心要点
1. 使用双向高质量 teacher 蒸馏出因果 student，保持生成质量同时支持流式推理
2. 核心挑战：训练时 history 来自 GT，推理时 history 来自模型自身输出，存在 distribution shift
3. RAVEN 等工作通过 Training-time Test 和 CM-GRPO 解决此 distribution gap

## 代表工作
- [[RAVEN]]: 在 CausVid 基础上引入在线 RL 对齐，解决 training-inference distribution gap

## 相关概念
- [[Diffusion Model]]（基础生成模型）
- [[DMD]]（Diffusion Model Distillation，蒸馏加速）
- [[World Model]]（视频生成作为世界模型的一种形式）

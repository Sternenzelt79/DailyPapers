---
type: concept
aliases: [RSSM, Recurrent State Space Model]
---

# RSSM

## 定义
Recurrent State Space Model，DreamerV3 等 model-based RL 方法的核心世界模型架构，将 latent state 分解为确定性（GRU/RNN 记忆）和随机性（VAE 采样）两部分，支持在 latent space 中高效进行长时序想象（imagination）。

## 数学形式
$$h_t = f_\phi(h_{t-1}, z_{t-1}, a_{t-1}) \quad \text{(deterministic)}$$
$$z_t \sim q_\phi(z_t \mid h_t, x_t) \quad \text{(stochastic, posterior)}$$
$$\hat{z}_t \sim p_\phi(\hat{z}_t \mid h_t) \quad \text{(stochastic, prior)}$$

## 核心要点
1. 确定性路径（GRU）保持长时依赖；随机路径（VAE）捕捉不确定性
2. 在 latent space 中做 imagination：不需要解码到图像空间
3. 支持 gradient-based 和 model-free 策略学习
4. GPLD 在 RSSM 转移函数上加梯度惩罚，强制局部平滑性

## 代表工作
- [[DreamerV3]]: RSSM 是其核心世界模型组件
- [[GPLD]]: 在 RSSM 基础上添加 Jacobian 梯度惩罚正则项

## 相关概念
- [[DreamerV3]]
- [[状态空间模型]]
- [[扩散世界模型]]

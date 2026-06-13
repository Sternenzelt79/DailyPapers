---
type: concept
aliases: [SVD, Stable Video Diffusion, 稳定视频扩散]
---

# Stable Video Diffusion（SVD）

## 定义

SVD 是 Stability AI 发布的视频生成扩散模型（1.5B 参数），以单帧图像为条件生成连续视频帧，在机器人操作领域被广泛微调为**世界模型**，用于预测未来视觉状态。

## 数学形式

训练目标（条件视频扩散）：

$$
\mathcal{L}_{wm} = \mathbb{E}_{(I,L,x_0) \sim \mathcal{D}, \varepsilon, \tau}\!\left[\left\|\mathcal{W}_\phi(x_\tau, \tau, I, L) - x_0\right\|_2^2\right]
$$

其中 $I$ 为当前帧，$L$ 为语言条件，$x_0$ 为目标未来帧潜变量。

## 核心要点

1. **1.5B 参数**：比图像扩散模型更大，具备时序建模能力；
2. **U-Net 架构**：多层解码器特征包含不同尺度的时序信息，适合 FPN 聚合；
3. **部分去噪技巧**：仅做 1 步去噪即可提取语义丰富的多尺度潜特征，无需完整生成视频；
4. **机器人操作微调**：MemoryVLA++ 在具体任务数据上微调 20k-40k 步；
5. **记忆引导使用**：用历史记忆 token 过滤想象 token，保留决策相关预测。

## 代表工作

- [[MemoryVLApp]]: 微调 SVD 作为操作世界模型，提取 FPN 多尺度特征用于动作预测

## 相关概念

- [[扩散模型]]
- [[FPN]]
- [[世界模型]]
- [[DDIM]]

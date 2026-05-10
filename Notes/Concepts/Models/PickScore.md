---
type: concept
aliases: [Pick-a-Pic Score, PickScore Reward]
---

# PickScore

## 定义
基于 Pick-a-Pic 人类偏好数据集训练的图像质量评估模型，通过对比学习估计图像与文字 prompt 的对齐质量，常用作扩散模型 RL 训练的奖励信号。

## 数学形式
$$
r_{\text{Pick}}(\mathbf{x}, c) = \text{sigmoid}\left(\text{CLIP}_\text{img}(\mathbf{x}) \cdot \text{CLIP}_\text{txt}(c)^T / \tau\right)
$$

对比学习框架，在 500k+ 人类偏好对上训练。

## 核心要点
1. 基于 Pick-a-Pic 数据集，包含真实用户在两张图之间的选择偏好
2. 与 [[ImageReward]] 互补：PickScore 在审美和多样性方面更强，ImageReward 在无害性方面更强
3. 常作为 T2I 生成质量的标准评估指标
4. [[MARBLE]]、[[CDM]] 等工作用其评估 diffusion 训练质量

## 代表工作
- [[MARBLE]]: 多奖励 RL 中用 PickScore 作为奖励之一
- [[CDM]]: 用 PickScore 评估蒸馏模型的生成质量

## 相关概念
- [[ImageReward]]
- [[Diffusion Model]]
- [[MARBLE]]

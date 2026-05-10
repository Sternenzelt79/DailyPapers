---
type: concept
aliases: [Image Reward Model, ImageReward Score]
---

# ImageReward

## 定义
一种基于人类偏好数据训练的图像质量奖励模型，输入图像+文字 prompt，输出标量奖励分数，用于扩散模型的 RL fine-tuning。

## 数学形式
$$
r_{\text{ImageReward}}(\mathbf{x}, c) = \text{MLP}\left(\text{CLIP}(\mathbf{x}, c)\right) \cdot w^T
$$

基于 CLIP 特征 + MLP 回归头，在人工标注的偏好数据上训练。

## 核心要点
1. 第一个专门为 text-to-image 设计的人类偏好奖励模型
2. 覆盖美学、文本对齐、无害性多个维度
3. 常与 [[PickScore]] 对比，后者在某些 benchmark 上更好
4. [[MARBLE]] 等工作将其作为多奖励之一用于 diffusion RL fine-tuning

## 代表工作
- [[MARBLE]]: 将 ImageReward 作为多奖励组件之一进行梯度调和

## 相关概念
- [[PickScore]]
- [[Diffusion Model]]
- [[MARBLE]]

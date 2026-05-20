---
type: concept
aliases: [Direct Preference Optimization, 直接偏好优化]
---

# DPO

## 定义
一种无需显式奖励模型的偏好对齐方法，直接用人类偏好数据优化语言模型，将 RLHF 转化为分类问题。

## 数学形式
$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)} \left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

其中 $y_w$ 为 preferred 样本，$y_l$ 为 rejected 样本，$\beta$ 为温度系数。

## 核心要点
1. 绕开显式奖励模型，直接从 preferred/rejected 对优化策略
2. 将 RL 目标转化为加权最大似然，训练稳定
3. 需要预训练参考模型 $\pi_{\text{ref}}$ 作为 KL 正则锚点
4. 在 VLA 中的应用：用成功/失败的执行轨迹构建偏好对（见 [[DEFLECT]]、[[COAST]]）

## 代表工作
- Rafailov et al. (2023): Direct Preference Optimization: Your Language Model is Secretly a Reward Model

## 相关概念
- [[RLHF]]
- [[GRPO]]
- [[SFT]]
- [[Activation Steering]]
